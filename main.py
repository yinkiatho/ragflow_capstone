import asyncio
import sys
import signal
import numexpr as ne
from ragflow_python.testing_bot.testing_goldens_mak import run_test
from dotenv import load_dotenv

# Set to desired number of threads
ne.set_num_threads(16)

# Fix for aiodns on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    load_dotenv()

    loop = asyncio.get_event_loop()

    async def graceful_shutdown(sig, loop):
        print(f"Received exit signal {sig.name}, shutting down gracefully...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    # Signal handling for UNIX-like systems
    if sys.platform != 'win32':  
        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(graceful_shutdown(sig, loop)))
    else:
        print("Signal handlers are not supported on Windows, use CTRL+C to stop.")

    try:
        loop.run_until_complete(run_test(generate_attacks=False, fetch_chunks=False,
                                         activate_defense=False))
    except KeyboardInterrupt:
        print("Manual interruption, shutting down...")
    finally:
        loop.close()


if __name__ == '__main__':
    main()

