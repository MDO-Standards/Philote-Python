"""
Explicit discipline server for Julia disciplines.
"""
from concurrent import futures
import grpc

from .wrapper import JuliaWrapperDiscipline
from .config import PhiloteConfig
import philote_mdo.general as pmdo


def serve_explicit_discipline(config: PhiloteConfig):
    """
    Start a gRPC server hosting an explicit Julia discipline.

    Args:
        config: PhiloteConfig object with discipline and server configuration
    """
    print("=" * 70)
    print("  Philote Julia Server (Python wrapper + juliacall)")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Julia file:  {config.discipline.julia_file}")
    print(f"  Julia type:  {config.discipline.julia_type}")
    print(f"  Server addr: {config.server.address}")
    print(f"  Max workers: {config.server.max_workers}")
    if config.discipline.options:
        print(f"  Options:     {config.discipline.options}")
    print()

    # Create the wrapper discipline
    discipline_wrapper = JuliaWrapperDiscipline(
        julia_file=config.discipline.julia_file,
        julia_type=config.discipline.julia_type,
        options=config.discipline.options
    )

    print()

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.server.max_workers))

    # Create discipline server and attach to gRPC server
    discipline_server = pmdo.ExplicitServer(discipline=discipline_wrapper)
    discipline_server.attach_to_server(server)

    # Start server
    server.add_insecure_port(config.server.address)
    server.start()
    print(f"✓ Server started successfully!")
    print(f"  Listening on: {config.server.address}")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop(grace=2.0)
        print("Server stopped.")
