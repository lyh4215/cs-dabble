
import asyncio
import os
import time

HOST = "127.0.0.1"
PORT = 9000

WORKERS = 4
SERVICE_TIME = 0.05  # 50 ms per request
QUEUE_LIMIT = int(os.getenv("QUEUE_LIMIT", "0"))

queue = asyncio.Queue(maxsize=QUEUE_LIMIT)

completed = 0
rejected = 0
peak_queue = 0


async def worker(worker_id):
    global completed

    while True:
        future, enqueued_at = await queue.get()

        try:
            # Simulated network service processing.
            await asyncio.sleep(SERVICE_TIME)

            completed += 1
            waiting_ms = (
                time.perf_counter() - enqueued_at
                - SERVICE_TIME
            ) * 1000

            if not future.done():
                future.set_result(max(waiting_ms, 0))

        finally:
            queue.task_done()


async def handle_client(reader, writer):
    global rejected, peak_queue

    loop = asyncio.get_running_loop()

    try:
        data = await reader.readline()
        if not data:
            return

        future = loop.create_future()

        try:
            queue.put_nowait(
                (future, time.perf_counter())
            )
            peak_queue = max(peak_queue, queue.qsize())

        except asyncio.QueueFull:
            rejected += 1
            writer.write(b"BUSY\n")
            await writer.drain()
            return

        waiting_ms = await future

        writer.write(
            f"OK {waiting_ms:.3f}\n".encode()
        )
        await writer.drain()

    finally:
        writer.close()
        await writer.wait_closed()


async def monitor():
    while True:
        await asyncio.sleep(1)

        print(
            f"queue={queue.qsize():4d} "
            f"peak={peak_queue:4d} "
            f"completed={completed:5d} "
            f"rejected={rejected:5d}"
        )


async def main():
    for i in range(WORKERS):
        asyncio.create_task(worker(i))

    asyncio.create_task(monitor())

    server = await asyncio.start_server(
        handle_client,
        HOST,
        PORT,
        limit=4096,
    )

    print(
        f"Listening on {HOST}:{PORT} | "
        f"workers={WORKERS}, "
        f"service={SERVICE_TIME*1000:.0f}ms, "
        f"queue_limit={QUEUE_LIMIT or 'unbounded'}"
    )

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
