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
Shared helpers for the test suite.

Two things changed when servers gained per-job state, and between them they
account for nearly every test that had to be touched.

First, RPC handlers now read the ``philote-job-id`` metadata header, so a
handler called directly needs a context that can produce one. A bare
``Mock()`` cannot: ``invocation_metadata()`` returns another ``Mock``, which is
not iterable. :func:`job_context` supplies one that works.

Second, a server no longer owns a discipline. It builds one per job, so a test
that used to configure ``server._discipline`` now configures
``job.discipline``. :func:`make_server` sets both up in one call.
"""
from unittest.mock import Mock, patch

from philote_mdo.general.job import JOB_METADATA_KEY


def job_context(job=None, job_id=None):
    """
    Returns a fake ``ServicerContext`` naming a job.

    ``abort`` is left as a plain ``Mock``, so it records the call and returns
    instead of raising the way real gRPC does. The suite depends on that: tests
    assert ``context.abort.assert_called_once()`` and then go on to inspect
    what the handler produced.

    Parameters
    ----------
    job : Job, optional
        Job to name. Takes precedence over ``job_id``.
    job_id : str, optional
        Job id to name, when there is no job object to hand.

    Returns
    -------
    unittest.mock.Mock
        Context whose ``invocation_metadata()`` carries the job header.
    """
    if job is not None:
        job_id = job.job_id

    context = Mock()

    if job_id is None:
        context.invocation_metadata.return_value = ()
    else:
        context.invocation_metadata.return_value = ((JOB_METADATA_KEY, job_id),)

    return context


def make_server(server_cls, discipline_factory, **kwargs):
    """
    Builds a server, starts one job on it, and returns a context for that job.

    Parameters
    ----------
    server_cls : type
        ``DisciplineServer`` or one of its subclasses.
    discipline_factory : callable
        Zero-argument callable returning a fresh discipline.
    **kwargs
        Passed through to the server constructor. ``ttl`` defaults to ``None``
        so that no sweeper thread is started for a unit test.

    Returns
    -------
    tuple
        ``(server, job, context)``. Configure the discipline through
        ``job.discipline``.
    """
    kwargs.setdefault("ttl", None)

    server = server_cls(discipline_factory=discipline_factory, **kwargs)
    job = server._jobs.create()

    return server, job, job_context(job=job)


def bind_job(client, job_id="test-job"):
    """
    Binds a client to a job id without making an RPC.

    Client-side unit tests patch only the service stub they are exercising, so
    the real ``DisciplineServiceStub`` sits on an intercepted ``Mock`` channel.
    Letting the lazy start fire there would attempt a genuine ``StartJob``.
    Presetting the id keeps those tests on the subject they are testing.

    Parameters
    ----------
    client : DisciplineClient
        The client to bind.
    job_id : str, optional
        Id to bind to.

    Returns
    -------
    DisciplineClient
        The same client, for chaining.
    """
    client._job_id = job_id
    return client


def patch_discipline_stub(job_id="test-job"):
    """
    Replaces ``DisciplineServiceStub`` with a mock for client-side unit tests.

    Two things make this necessary. Clients now wrap their channel in an
    interceptor, so a ``Mock`` channel is no longer inert: gRPC's real
    machinery runs on top of it and fails when it tries to unpack a mock
    response. And components that claim a job from inside their constructor --
    the OpenMDAO bindings call ``send_options`` in ``__init__`` -- leave no
    window to bind an id afterwards.

    Patching the base stub covers every call that goes through it, including
    the lazy ``StartJob``.

    Parameters
    ----------
    job_id : str, optional
        Id that ``StartJob`` appears to return.

    Returns
    -------
    unittest.mock._patch
        Start it with ``.start()`` and stop it with ``.stop()``, or use it as
        a context manager.
    """
    patcher = patch(
        "philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub"
    )
    real_start = patcher.start
    real_stop = patcher.stop

    def start():
        stub_cls = real_start()
        stub_cls.return_value.StartJob.return_value = Mock(job_id=job_id)
        return stub_cls

    patcher.start = start
    patcher.stop = real_stop

    return patcher


def make_server_from_instance(server_cls, discipline, **kwargs):
    """
    Builds a server whose factory always hands back one given instance.

    Isolation between jobs is the point of the factory, so production code
    should never do this. It is useful in a unit test that configures a
    discipline by hand and then calls the job-independent RPCs -- ``GetInfo``
    and ``GetAvailableOptions`` answer from an instance of the server's own,
    and this makes that instance the same object the test configured.

    Parameters
    ----------
    server_cls : type
        ``DisciplineServer`` or one of its subclasses.
    discipline : Discipline
        The instance every job and every describe() call will receive.
    **kwargs
        Passed through to the server constructor.

    Returns
    -------
    tuple
        ``(server, job, context)``.
    """
    return make_server(server_cls, lambda: discipline, **kwargs)


class Aborted(Exception):
    """
    Stands in for the exception real gRPC raises out of ``context.abort``.
    """


def aborting_job_context(job=None, job_id=None):
    """
    Returns a context whose ``abort`` raises, as real gRPC's does.

    :func:`job_context` leaves ``abort`` as a plain mock because most of the
    suite asserts on it and then carries on inspecting the handler's output.
    That is the wrong shape for testing an abort path itself: without an
    exception the handler runs on past the abort and fails a second time
    further down, so the status code under test gets buried.

    Parameters
    ----------
    job : Job, optional
        Job to name.
    job_id : str, optional
        Job id to name.

    Returns
    -------
    unittest.mock.Mock
        Context that raises :class:`Aborted` when the handler aborts.
    """
    context = job_context(job=job, job_id=job_id)
    context.abort.side_effect = Aborted

    return context
