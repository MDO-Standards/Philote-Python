(tutorials:quick_start)=
# Quick Start

Client/server interactions might seem difficult when starting out, but the
purpose of this library is to help out and abstract away the difficult bits.
This quick start guide is intended to familiarize you with the basic operating
principles of the Philote MDO standard and get you started using this Python
implementation.

:::{note}
This guide attempts to be as user friendly as possible. However, it is likely
that some basic understanding of Multidisciplinary Design Analysis and
Optimization may be necessary (e.g., what a discipline is, etc.).
:::


## Disciplines

Before we take a closer look at clients and servers, we need a discipline that
we will attach to a server and call from the client. Philote-Python implements
disciplines in a similar way to OpenMDAO. We create a class, inheriting from a
base class and then specialize some methods to run the calculations we want.

To illustrate this, let us take a look at a simple paraboloid problem (the same
problem as in the OpenMDAO documentation):

\begin{align}
f(x,y) &= (x-3)^2 + x y + (y+4)^2 - 3
\end{align}

To create a discipline that executes this equation, we create a class and
inherit from the ExplicitDiscipline class Philote-Python provides:

:::{code-block} python
class Paraboloid(pmdo.ExplicitDiscipline):
    """
    Basic two-dimensional paraboloid example (explicit) discipline.
    """

    def setup(self):
        self.add_input("x", shape=(1,), units="m")
        self.add_input("y", shape=(1,), units="m")

        self.add_output("f_xy", shape=(1,), units="m**2")

    def setup_partials(self):
        self.declare_partials("f_xy", "x")
        self.declare_partials("f_xy", "y")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]

        outputs["f_xy"] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs["x"]
        y = inputs["y"]

        partials["f_xy", "x"] = 2.0 * x - 6.0 + y
        partials["f_xy", "y"] = 2.0 * y + 8.0 + x
:::

How to declare and implement this (explicit) discipline is discussed in
{ref}`tutorials:explicit`. For the time being, we will skip over the
discipline member functions or what the do in detail.

Now, the discipline must first be attached to a server to run.


## Standing Up an Analysis Server

The discipline created in the previous section serves as the implementation for
the analysis server we will stand up in this section. Philote-Python attempts to
abstract away most aspects of using gRPC, so that discipline developers can
focus on their respective fields, rather than client-server communication.
Because of this, a server class is provided, to which analysis disciplines can
be attached.

First, a gRPC channel needs to be generated:

:::{code-block} python
from concurrent import futures
import grpc
# ...

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
:::

Next, the **Paraboloid** discipline is attached to the server:

:::{code-block} python
import philote_mdo.general as pmdo
from philote_mdo.examples import Paraboloid
# ...

discipline = pmdo.ExplicitServer(discipline=Paraboloid())
discipline.attach_to_server(server)
:::

Finally, the port of the server is defined (opening a port is necessary for
network communication) and the server is started:

:::{code-block} python
server.add_insecure_port("[::]:50051")
server.start()
print("Server started. Listening on port 50051.")
server.wait_for_termination()
:::

In this example, the server waits for a termination signal. Using *Ctrl-C* will
kill the server (on Unix and Unix-like systems) when it is no longer needed.

:::{warning}
This example uses an insecure port. Production environments should
generally always use encrypted network traffic to minimize security
vulnerabilities and third parties snooping on data exchanged. The code presented
here is a tutorial and not intended for production use.
:::


The above code snippets were taken slightly out of order for didactical
purposes. For completeness, here is the full example server
(paraboloid_explicit.py):

:::{code-block} python
from concurrent import futures
import grpc
import philote_mdo.general as pmdo
from philote_mdo.examples import Paraboloid


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

discipline = pmdo.ExplicitServer(discipline=Paraboloid())
discipline.attach_to_server(server)

server.add_insecure_port("[::]:50051")
server.start()
print("Server started. Listening on port 50051.")
server.wait_for_termination()
:::


## Calling the Discipline Using a Client

Now that a server is running, it can be querried using a client. Philote-Python
offers a number of clients for this purpose, ranging from the general
implementation to OpenMDAO and CSDL components. However, under the hood, the
OpenMDAO and CSDL components use the general client implementation.

This example will use the general explicit client implementation. Despite it
being fully functional and able to run in scientific workflows, it probably is
not a realistic MDO workflow. {ref}`tutorials:openmdao` and
{ref}`tutorials:csdl` demonstrate calling Philote disciplines from the OpenMDAO
and CSDL MDO frameworks.

First, the explicit client must be imported and initialized using a gRPC
channel. 

:::{code-block}
import grpc
import numpy as np
from philote_mdo.general import ExplicitClient


client = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))
:::

:::{warning}
The same disclaimer as above applies here. It is not recommended using an
insecure channel for production work. Generally, all production work should be
encrypted.
:::

Now the client should be connected to the analysis server. While it may be
tempting to just call the function evaluation at this point, hold your horses.
There are a few necessary steps that are mandatory to ensure proper behavior.
The first is to sync the stream options between the client and server (details
of what happens here can be found in **Setting up the Connection**).

:::{code-block}
# ...

# transfer the stream options to the server
client.send_stream_options()

# ...
:::

Next, the setup function must be run, after which the variable and partials meta
data must be retrieved from the server:

:::{code-block}
# ...

# run setup
client.run_setup()
client.get_variable_definitions()
client.get_partials_definitions()

# ...
:::

Now, the client is ready to call the function evaluation. An input dictionary is
defined using the variable name as the key and numpy arrays (size=1) and passed
to the **run_compute** function.

:::{code-block}
# ...

# define some inputs
inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
outputs = {}

# run a function evaluation
outputs = client.run_compute(inputs)

print(outputs)
:::

The script should now have printed out the outputs dictionary. Congratulations,
you have run your first client-server analysis using Philote/Philote-Python!


The full client script (paraboloid_client.py) is:

:::{code-block}
import grpc
import numpy as np
from philote_mdo.general import ExplicitClient


client = ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

# transfer the stream options to the server
client.send_stream_options()

# run setup
client.run_setup()
client.get_variable_definitions()
client.get_partials_definitions()

# define some inputs
inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
outputs = {}

# run a function evaluation
outputs = client.run_compute(inputs)

print(outputs)
:::

## Potential Pitfalls

Keep in mind that Philote requires network communication. For the tutorials it
is probably best to run both the server and client on the same machine. However,
it is possible to call a remote analysis discipline that is separate of the
client. This requires opening the appropriate port (in this case, *50051*) and
managing permissions. Many of these issues can be avoided by running the
examples on the same machine.