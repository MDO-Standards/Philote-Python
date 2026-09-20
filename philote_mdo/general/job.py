# Philote-Python
#
# Copyright 2022-2025 Christopher A. Lupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This work has been cleared for public release, distribution unlimited, case
# number: AFRL-2023-5713.
#
# The views expressed are those of the authors and do not reflect the
# official guidance or position of the United States Government, the
# Department of Defense or of the United States Air Force.
#
# Statement from DoD: The Appearance of external hyperlinks does not
# constitute endorsement by the United States Department of Defense (DoD) of
# the linked websites, of the information, products, or services contained
# therein. The DoD does not exercise any editorial, security, or other
# control over the information you may find at these locations.
"""
Server-side job sessions.

A Philote server used to hold one discipline instance and share it across
every client that connected. Because the protocol has each client run a setup
sequence that mutates that instance -- ``SetOptions``, ``Setup``, and the
in-place shape edits of ``SetVariableShapes`` -- two concurrent clients
corrupted one another.

A :class:`Job` is a session that owns its own discipline instance, so the state
each client builds is private to it. Clients get a job id from ``StartJob`` and
present it in the ``philote-job-id`` metadata header on every later call.

Jobs are session-shaped rather than request-shaped: there are few of them, each
lives across many evaluations, and each can hold something substantial such as
an initialised solver or an ``om.Problem``. That is why they are capped and
swept rather than allowed to accumulate.
"""
import threading
import time
import uuid

import philote_mdo.generated.data_pb2 as data
from philote_mdo.utils.validation import (
    JobCapacityError,
    JobNotFoundError,
    JobStateError,
)


# metadata header carrying the job id on every job-scoped RPC
JOB_METADATA_KEY = "philote-job-id"

# doubles per message, matching the server default. Held per job because
# SetStreamOptions is a per-client setting.
DEFAULT_NUM_DOUBLE = 100000

# a job that has gone unused for this many seconds is evicted
DEFAULT_TTL = 3600.0

# how often the sweeper thread looks for expired jobs
DEFAULT_SWEEP_INTERVAL = 60.0

# concurrent jobs allowed per server. Deliberately modest: each job holds a
# discipline instance, and the gRPC thread pool caps useful concurrency anyway.
DEFAULT_MAX_JOBS = 8


class JobState:
    """The stages a job moves through, in order."""

    NEW = "new"
    SETUP = "setup"
    READY = "ready"
    CLOSED = "closed"


class Job:
    """One client session, owning one discipline instance.

    Parameters
    ----------
    job_id : str
        Server-assigned identifier, opaque to the client.
    discipline : Discipline or None
        The discipline instance this job owns. ``None`` only while
        :meth:`JobStore.create` is still building it.
    """

    def __init__(self, job_id, discipline=None):
        self.job_id = job_id
        self.discipline = discipline

        # stream options are per client, so they live here rather than on the
        # server that all clients share
        self.stream_opts = data.StreamOptions(num_double=DEFAULT_NUM_DOUBLE)

        self.state = JobState.NEW

        # serialises calls within this job. Separate jobs never contend on it,
        # so two clients can be inside compute() at the same time.
        self.lock = threading.Lock()

        self.last_used = time.monotonic()

    def touch(self):
        """Marks the job as still in use, deferring eviction."""
        self.last_used = time.monotonic()

    def require_before_setup(self, rpc_name):
        """Rejects a call that must precede ``Setup`` but arrived after it.

        Parameters
        ----------
        rpc_name : str
            Name of the RPC, used in the error message.

        Raises
        ------
        JobStateError
            If ``Setup`` has already run for this job.
        """
        if self.state in (JobState.SETUP, JobState.READY):
            raise JobStateError(
                f"{rpc_name}: job '{self.job_id}' has already run Setup. "
                f"The variable metadata was built from the previous values, "
                f"so start a new job instead."
            )

    def __repr__(self):
        return f"<Job {self.job_id} state={self.state}>"


class JobStore:
    """Holds the live jobs for one server and enforces their limits.

    Parameters
    ----------
    discipline_factory : callable
        Zero-argument callable returning a fresh ``Discipline``. A class works
        directly when its ``initialize()`` does its own configuration; a
        discipline configured from outside needs a closure or
        ``functools.partial``.
    max_jobs : int, optional
        Concurrent jobs allowed before ``StartJob`` is refused.
    ttl : float or None, optional
        Seconds a job may sit unused before eviction. ``None`` disables both
        expiry and the sweeper thread.
    sweep_interval : float, optional
        Seconds between sweeps. Ignored when ``ttl`` is ``None``.
    """

    def __init__(
        self,
        discipline_factory,
        max_jobs=DEFAULT_MAX_JOBS,
        ttl=DEFAULT_TTL,
        sweep_interval=DEFAULT_SWEEP_INTERVAL,
    ):
        if not callable(discipline_factory):
            raise TypeError(
                f"discipline must be a zero-argument callable returning a "
                f"Discipline, got an instance of "
                f"{type(discipline_factory).__name__}. Pass the class rather "
                f"than an instance of it -- "
                f"ExplicitServer(discipline={type(discipline_factory).__name__})"
                f", not "
                f"ExplicitServer(discipline={type(discipline_factory).__name__}())"
                f". The server builds one discipline per job, so it needs "
                f"something it can call."
            )

        self._factory = discipline_factory
        self._max_jobs = max_jobs
        self._ttl = ttl
        self._jobs = {}
        self._lock = threading.Lock()

        # lazily built instance answering the job-independent RPCs
        self._prototype = None

        self._stop = threading.Event()
        self._sweeper = None

        if ttl is not None:
            self._sweeper = threading.Thread(
                target=self._sweep_loop,
                args=(sweep_interval,),
                name="philote-job-sweeper",
                daemon=True,
            )
            self._sweeper.start()

    @property
    def max_jobs(self):
        return self._max_jobs

    def describe(self):
        """Returns an instance for the RPCs that do not belong to a job.

        ``GetInfo`` and ``GetAvailableOptions`` report properties of the
        discipline class -- its name, version, and the option schema built by
        ``initialize()`` -- so a client must be able to call them before it has
        a job. One instance is built on first use and reused, and it is never
        handed to a job, so nothing a client does can reach it.

        Returns
        -------
        Discipline
        """
        with self._lock:
            if self._prototype is None:
                self._prototype = self._factory()

            return self._prototype

    def create(self):
        """Starts a job and builds its discipline.

        Returns
        -------
        Job
            The new job, with its discipline built and back-linked.

        Raises
        ------
        JobCapacityError
            If the server already holds ``max_jobs`` jobs.
        """
        # reserve the slot under the store lock, then build the discipline
        # outside it. Construction can be slow -- loading a mesh, building an
        # om.Problem -- and holding the store lock through it would stall every
        # other job's calls.
        expired = []

        try:
            with self._lock:
                # reclaim expired slots first, so a dead client cannot make
                # StartJob fail while the sweeper has yet to run
                expired = self._expired_locked()

                for old in expired:
                    self._jobs.pop(old.job_id, None)

                if len(self._jobs) >= self._max_jobs:
                    raise JobCapacityError(
                        f"StartJob: server already holds its maximum of "
                        f"{self._max_jobs} jobs. End a job before starting "
                        f"another, or raise max_jobs."
                    )

                job_id = uuid.uuid4().hex
                job = Job(job_id)
                self._jobs[job_id] = job
        finally:
            # outside the lock, and on the capacity path too: whatever this
            # call evicted still has to release what it held
            for old in expired:
                self._teardown(old)

        try:
            discipline = self._factory()
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise

        job.discipline = discipline

        # let the discipline reach its own job, for the job id and, once the
        # file API lands, its scratch directory
        discipline.job = job

        return job

    def get(self, job_id):
        """Looks up a live job and marks it as used.

        Parameters
        ----------
        job_id : str
            The id from the ``philote-job-id`` header.

        Returns
        -------
        Job

        Raises
        ------
        JobNotFoundError
            If no such job exists, or it has been closed or evicted.
        """
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                raise JobNotFoundError(
                    f"job '{job_id}' is unknown to this server. It may have "
                    f"been ended, expired, or belonged to a server that has "
                    f"since restarted. Start a new job; any state it held is "
                    f"gone."
                )

            job.touch()

        return job

    def close(self, job_id):
        """Ends a job and releases whatever its discipline holds.

        Parameters
        ----------
        job_id : str
            The job to end.

        Raises
        ------
        JobNotFoundError
            If no such job exists.
        """
        with self._lock:
            job = self._jobs.pop(job_id, None)

        if job is None:
            raise JobNotFoundError(
                f"EndJob: job '{job_id}' is unknown to this server."
            )

        self._teardown(job)

    def sweep(self):
        """Evicts every job that has outlived the TTL.

        Called by the sweeper thread, and directly by tests.

        Returns
        -------
        list of str
            The ids that were evicted.
        """
        with self._lock:
            expired = self._expired_locked()

            for job in expired:
                self._jobs.pop(job.job_id, None)

        for job in expired:
            self._teardown(job)

        return [job.job_id for job in expired]

    def close_all(self):
        """Ends every job and stops the sweeper. Used on server shutdown."""
        self._stop.set()

        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()

        for job in jobs:
            self._teardown(job)

    def __len__(self):
        with self._lock:
            return len(self._jobs)

    def _expired_locked(self):
        """Returns the expired jobs. The store lock must be held."""
        if self._ttl is None:
            return []

        cutoff = time.monotonic() - self._ttl

        return [job for job in self._jobs.values() if job.last_used < cutoff]

    def _teardown(self, job):
        """Lets the discipline release its resources, then marks it closed."""
        job.state = JobState.CLOSED

        discipline = job.discipline

        if discipline is None:
            return

        teardown = getattr(discipline, "teardown_job", None)

        if callable(teardown):
            teardown()

        job.discipline = None

    def _sweep_loop(self, interval):
        while not self._stop.wait(interval):
            try:
                self.sweep()
            except Exception:  # pragma: no cover - a sweep must never kill
                pass           # the thread; the next tick tries again
