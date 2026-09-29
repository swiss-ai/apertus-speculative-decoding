import asyncio, sys
LISTEN=int(sys.argv[1]); BACKENDS=[int(x) for x in sys.argv[2].split(",")]; n=[0]
async def pipe(r, w):
    try:
        while True:
            d=await r.read(1<<16)
            if not d: break
            w.write(d); await w.drain()
    except Exception: pass
    finally:
        try: w.close()
        except Exception: pass
async def handle(cr, cw):
    port=BACKENDS[n[0] % len(BACKENDS)]; n[0]+=1
    try: br, bw = await asyncio.open_connection("127.0.0.1", port)
    except Exception:
        cw.close(); return
    await asyncio.gather(pipe(cr, bw), pipe(br, cw))
async def main():
    s=await asyncio.start_server(handle, "127.0.0.1", LISTEN)
    async with s: await s.serve_forever()
asyncio.run(main())
