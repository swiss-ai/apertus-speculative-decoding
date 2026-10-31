# Bounded read-only core metadata

Analysis stopped after **355,356 cumulative bytes** read from the four core
files. The limit was 4 MiB. No full-core scan/hash/copy, GPU reexecution,
debugger core session, wave/register analysis, kernel-argument read or
environment dump was performed. The core paths, sizes and nanosecond mtimes
are preserved in `headers.json`; each read range has an offset, byte count,
raw metadata bytes and SHA256 in the corresponding phase artifact.

| Phase | Core bytes read |
|---|---:|
| ELF/program headers and note headers/names | 183,462 |
| Two AMDGPU notes and two CPU NT_FILE notes | 148,646 |
| Queue indices, three candidate packets and loader header | 496 |
| Bounded queue window, descriptors and loader link_map/URI metadata | 22,624 |
| Final two 64-byte descriptor checks | 128 |
| **Total** | **355,356** |

Both GPU cores are ELF64 little-endian AMDGPU core files. Each has one
AMDGPU note of type 33 and size 920 bytes. An observed eight-byte prefix
contains words `(1, 0)`; the subsequent KFD 1.17 header describes 16 bytes
of runtime information, four 120-byte agent records and six 64-byte queue
records, exactly accounting for the descriptor. The pinned
[Linux KFD 1.17 header](https://raw.githubusercontent.com/torvalds/linux/d78f0ee0406803cda8801fd5201746ccf89e5e4a/include/uapi/linux/kfd_ioctl.h)
and its hash are retained locally. The note type and general snapshot format
are documented in the [AMD core-file guide](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.2.1/LLVM/llvm/html/AMDGPUUsage.html#core-file-notes).

Each process has exactly one queue with exception status `0x8`: queue 1,
COMPUTE_AQL, on GPU IDs 53900 and 18385. The matching header and exact-image
hsakmttypes enum identify this as the illegal-instruction bit. Other queue
exception statuses are zero. Runtime metadata has `ttmp_setup=0`.

The fault ring bases are `0x14be1ca00000` and `0x1500aec00000`, each
`0x200000` above its ROCm abort-queue log address. This numerical relationship
does not establish an internal queue role. Read/write indices were
1291/1374 and 1275/1358. Headers at read and read−1 were INVALID; the latest
submitted packet was BARRIER_AND. There is no uniquely established active
dispatch position.

A bounded window of at most 128 packets per process yielded 47 pending or
retained dispatch candidates. Loader metadata validated `r_debug` version 11
and yielded 22 code-object URIs per process. These are candidates, not a
faulting kernel or instruction PC. No symbol identity was established.

The raw `dispatch-map.json` includes a provisional V3+ descriptor decode at
offset 16. It returned zero offsets throughout. Those fields
`kernel_entry_offset` and `kernel_entry_address` are **unvalidated and rejected
as findings**. Descriptor bytes are nonzero; a final 64-byte comparison found
no saved CPU mapping for the sampled address and identical GPU bytes across
ranks. It did not validate the assumed layout. No further reads were made.

Tool versions are retained in `notes.json`: host readelf 2.43.1 and ROCgdb
15.2 (`rocm-rel-6.3-39`); ROCgdb was invoked only for `--version`. Exact image
and host headers, pinned primary-source metadata, extraction scripts, read
ranges and SHA256 manifests are retained. CPU NT_FILE mappings contain file
mapping metadata, not process memory or environment contents.
