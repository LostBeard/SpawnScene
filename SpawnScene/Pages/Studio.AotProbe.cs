using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnScene.Pages;

// ?autotest=aotprobe&probe=N - one host->device upload shape per page load, to find which one the AOT build traps on
// ("RuntimeError: function signature mismatch" in ArrayViewExtensions.CopyFromCPU<int>(ArrayView1D<int, Dense>,
// AcceleratorStream, int[]), reached from the trainer gate's scan stage, 2026-10-07). A wasm trap cannot be caught,
// so each probe runs alone and announces itself first.
public partial class Studio
{
    static void ProbeKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> v) => v[i] = i;

    /// <summary>A launcher-shaped generic sink (first parameter is the bound target).</summary>
    static void ProbeLauncherSink<TBody>(object target, AcceleratorStream stream, LongIndex1D n, TBody body) where TBody : struct { }

    /// <summary>A grid-stride body's shape: a struct holding an int view (as ILGPU.Algorithms' InitializerImplementation).</summary>
    struct ProbeBody { public ArrayView1D<int, Stride1D.Dense> View; public int Value; }

    static void ProbeBodyKernel(Index1D i, ProbeBody b) => b.View[i] = b.Value;

    /// <summary>InitializerImplementation's exact shape: generic over the element and the stride.</summary>
    readonly struct ProbeGenericGridBody<T, TStride> : IGridStrideKernelBody where T : unmanaged where TStride : struct, IStride1D
    {
        public ProbeGenericGridBody(ArrayView1D<T, TStride> view, T value) { View = view; Value = value; }
        public ArrayView1D<T, TStride> View { get; }
        public T Value { get; }
        public void Execute(LongIndex1D linearIndex) { if (linearIndex >= View.Length) return; View[linearIndex] = Value; }
        public void Finish() { }
    }

    /// <summary>The grid-stride body shape of ILGPU.Algorithms' InitializerImplementation, as our own type.</summary>
    readonly struct ProbeGridBody : IGridStrideKernelBody
    {
        public ProbeGridBody(ArrayView1D<int, Stride1D.Dense> view, int value) { View = view; Value = value; }
        public ArrayView1D<int, Stride1D.Dense> View { get; }
        public int Value { get; }
        public void Execute(LongIndex1D linearIndex) { if (linearIndex >= View.Length) return; View[linearIndex] = Value; }
        public void Finish() { }
    }

    static void ReplicaCopyHasNoData<T>(ArrayView1D<T, Stride1D.Dense> view, AcceleratorStream stream, T[] data)
        where T : unmanaged
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (view.HasNoData()) return;
        if (data.GetLength(0) < view.Extent.X) throw new ArgumentOutOfRangeException(nameof(data));
        view.BaseView.CopyFromCPU(stream, new ReadOnlySpan<T>(data));
    }

    static void ReplicaCopyDirect<T>(ArrayView1D<T, Stride1D.Dense> view, AcceleratorStream stream, T[] data)
        where T : unmanaged
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (!view.IsValid || view.Length < 1) return;
        if (data.GetLength(0) < view.Extent.X) throw new ArgumentOutOfRangeException(nameof(data));
        view.BaseView.CopyFromCPU(stream, new ReadOnlySpan<T>(data));
    }

    private async Task RunAotProbeAsync(int probe)
    {
        var accel = _gpuService.WebGPUAccelerator;
        Console.WriteLine($"[AotProbe] START {probe}");
        try
        {
            switch (probe)
            {
                case 0: // the gate's shape: Allocate1D(int[]) from a LINQ-built array, 4095 elements
                {
                    var a = Enumerable.Range(0, 4095).Select(i => i & 1).ToArray();
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 1: // a plain new int[4095]
                {
                    var a = new int[4095];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 2: // a small new int[16]
                {
                    var a = new int[16];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 3: // float[4095]
                {
                    var a = new float[4095];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                case 4: // Allocate1D<int>(n) then View.CopyFromCPU(int[]) (no stream overload)
                {
                    var a = new int[4095];
                    using var b = accel.Allocate1D<int>(a.Length);
                    b.View.CopyFromCPU(a);
                    break;
                }
                case 5: // the base view from a span, as CopyFromCPU does inside
                {
                    var a = new int[4095];
                    using var b = accel.Allocate1D<int>(a.Length);
                    b.View.BaseView.CopyFromCPU(accel.DefaultStream, new ReadOnlySpan<int>(a));
                    break;
                }
                case 6: // the same call through the Accelerator static type (as GpuFeatureDetector)
                {
                    Accelerator acc = accel;
                    var a = new int[4095];
                    using var b = acc.Allocate1D(a);
                    break;
                }
                case 8: // ORDER: uint[] first (as the radix sort gate does), then int[] in the same page
                {
                    using (var u = accel.Allocate1D(new uint[4095])) { }
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 8: uint[] done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 9: // ORDER: int[] first, then uint[]
                {
                    using (var i0 = accel.Allocate1D(new int[4095])) { }
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 9: int[] done, now uint[]");
                    using var b = accel.Allocate1D(new uint[4095]);
                    break;
                }
                case 10: // ORDER: float[] first, then int[]
                {
                    using (var f0 = accel.Allocate1D(new float[4095])) { }
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 10: float[] done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 11: // ORDER: the trainer gate's radix sort stage, then the scan stage's int[] upload
                {
                    bool ok = await RadixSortGateAsync();
                    Console.WriteLine($"[AotProbe] 11: radix sort gate {(ok ? "passed" : "failed")}, now int[]");
                    using var b = accel.Allocate1D(Enumerable.Range(0, 4095).Select(i => i & 1).ToArray());
                    break;
                }
                case 12: // ORDER: GpuRadixSort alone (one sort, uint keys/values), then int[]
                {
                    var keys = new uint[1025]; var vals = new uint[1025];
                    for (int i = 0; i < keys.Length; i++) { keys[i] = (uint)((i * 2654435761u) >> 2); vals[i] = (uint)i; }
                    using var kBuf = accel.Allocate1D(keys);
                    using var vBuf = accel.Allocate1D(vals);
                    using var sorter = new Services.GpuRadixSort(accel.NativeAccelerator.NativeDevice!, accel.NativeAccelerator.Queue!, accel);
                    sorter.EnsureCapacity(keys.Length);
                    sorter.Sort(kBuf.GetGPUBuffer()!, vBuf.GetGPUBuffer()!, keys.Length, 30);
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 12: one GpuRadixSort done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 13: // the scan stage's exact prelude: CreateScan, temp size, temp alloc, then Allocate1D(int[])
                {
                    var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddInt32>(ILGPU.Algorithms.ScanKind.Exclusive);
                    long tempLen = Math.Max(1L, accel.ComputeScanTempStorageSize<int>(4095));
                    using var shared = accel.Allocate1D<int>(tempLen);
                    Console.WriteLine("[AotProbe] 13: scan prelude done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 14: // only CreateScan, then Allocate1D(int[])
                {
                    var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddInt32>(ILGPU.Algorithms.ScanKind.Exclusive);
                    Console.WriteLine("[AotProbe] 14: CreateScan done, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 15: // only the temp size, then Allocate1D(int[])
                {
                    long tempLen = Math.Max(1L, accel.ComputeScanTempStorageSize<int>(4095));
                    Console.WriteLine($"[AotProbe] 15: temp size {tempLen}, now int[]");
                    using var b = accel.Allocate1D(new int[4095]);
                    break;
                }
                case 16: // CreateScan, then a REPLICA of ILGPU's CopyFromCPU<T>(ArrayView1D<T, Dense>, stream, T[]) - HasNoData
                {
                    var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddInt32>(ILGPU.Algorithms.ScanKind.Exclusive);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 16: CreateScan done, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 17: // the same replica with the struct's own members instead of the generic HasNoData<TView>
                {
                    var scan = accel.CreateScan<int, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddInt32>(ILGPU.Algorithms.ScanKind.Exclusive);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 17: CreateScan done, replica copy with direct members");
                    ReplicaCopyDirect(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 18: // load (compile) a trivial kernel over ArrayView1D<int, Dense>, no dispatch; then the replica copy
                {
                    var k = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(ProbeKernel);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 18: kernel loaded, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 19: // load AND dispatch it once; then the replica copy
                {
                    var k = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(ProbeKernel);
                    using var b = accel.Allocate1D<int>(4095);
                    k((Index1D)4095, b.View);
                    await accel.SynchronizeAsync();
                    Console.WriteLine("[AotProbe] 19: kernel dispatched, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 20: // reflection only: inflate HasNoData<ArrayView1D<int, Dense>> via MakeGenericMethod; then the replica copy
                {
                    var m = typeof(ArrayViewExtensions).GetMethods().First(mi => mi.Name == "HasNoData" && mi.IsGenericMethodDefinition)
                        .MakeGenericMethod(typeof(ArrayView1D<int, Stride1D.Dense>));
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] 20: inflated {m.Name}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 21: // a DynamicMethod taking ArrayView1D<int, Dense> (built, never run); then the replica copy
                {
                    var dm = new System.Reflection.Emit.DynamicMethod("probe", typeof(void),
                        new[] { typeof(ArrayView1D<int, Stride1D.Dense>) }, typeof(Studio).Module);
                    var il = dm.GetILGenerator();
                    il.Emit(System.Reflection.Emit.OpCodes.Ldarg_0);
                    il.Emit(System.Reflection.Emit.OpCodes.Box, typeof(ArrayView1D<int, Stride1D.Dense>));
                    il.Emit(System.Reflection.Emit.OpCodes.Pop);
                    il.Emit(System.Reflection.Emit.OpCodes.Ret);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 21: DynamicMethod built, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 22: // CreateInitializer<int, Dense> alone (CreateMultiPassScan's first line); then the replica copy
                {
                    var init = accel.CreateInitializer<int, Stride1D.Dense>();
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 22: initializer created, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 24: // CreateScan over FLOAT, then the INT replica copy - is the poisoning per element type?
                {
                    var scan = accel.CreateScan<float, Stride1D.Dense, Stride1D.Dense, ILGPU.Algorithms.ScanReduceOperations.AddFloat>(ILGPU.Algorithms.ScanKind.Exclusive);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 24: float CreateScan done, int replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 25: // ILGPU Interop.SizeOf(Type) of a struct holding an ArrayView1D<int, Dense> (reflection Unsafe.SizeOf<T>)
                {
                    int size = ILGPU.Interop.SizeOf(typeof(ProbeBody));
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] 25: SizeOf(ProbeBody) = {size}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 26: // the non-generic .NET 9+ RuntimeHelpers.SizeOf of the same struct - the proposed fix
                {
                    int size = System.Runtime.CompilerServices.RuntimeHelpers.SizeOf(typeof(ProbeBody).TypeHandle);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] 26: RuntimeHelpers.SizeOf(ProbeBody) = {size}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 27: // a NON-generic kernel whose parameter is a struct holding the int view; load only; then the replica
                {
                    var k = accel.LoadAutoGroupedStreamKernel<Index1D, ProbeBody>(ProbeBodyKernel);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 27: struct-param kernel loaded, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 28: // ILGPU.Algorithms' generic grid-stride kernel over our own body type; load only; then the replica
                {
                    var k = accel.LoadGridStrideKernel<ProbeGridBody>();
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 28: grid-stride kernel loaded, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 31: // grid-stride kernel over a GENERIC body struct instantiated <int, Dense> (InitializerImplementation's shape)
                {
                    var k = accel.LoadGridStrideKernel<ProbeGenericGridBody<int, Stride1D.Dense>>();
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine("[AotProbe] 31: generic-body grid-stride kernel loaded, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 32: // ILGPU frontend's way: Module.ResolveMethod(token, typeArgs, methodArgs) over the generic body's IL
                case 33: // the candidate fix: resolve open (no context), close via GetMemberWithSameMetadataDefinitionAs
                {
                    var closed = typeof(ProbeGenericGridBody<int, Stride1D.Dense>);
                    var exec = closed.GetMethod(nameof(IGridStrideKernelBody.Execute))!;
                    var il = exec.GetMethodBody()!.GetILAsByteArray()!;
                    int resolved = 0;
                    for (int i = 0; i + 4 < il.Length; i++)
                    {
                        if (il[i] != 0x28 && il[i] != 0x6F) continue;   // call / callvirt
                        int token = BitConverter.ToInt32(il, i + 1);
                        if ((token >> 24) is not (0x06 or 0x0A or 0x2B)) continue;   // MethodDef / MemberRef / MethodSpec
                        try
                        {
                            System.Reflection.MethodBase? m;
                            if (probe == 32) m = closed.Module.ResolveMethod(token, closed.GetGenericArguments(), null);
                            else
                            {
                                var open = closed.Module.ResolveMethod(token);
                                // Close the declaring type with the body's own arguments where it is generic over them.
                                var decl = open?.DeclaringType;
                                if (decl is { IsGenericTypeDefinition: true } && decl.GetGenericArguments().Length == 2)
                                    m = (System.Reflection.MethodBase?)decl.MakeGenericType(closed.GetGenericArguments())
                                        .GetMemberWithSameMetadataDefinitionAs(open!);
                                else m = open;
                            }
                            if (m != null) resolved++;
                        }
                        catch { }
                    }
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] {probe}: resolved {resolved} call tokens, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 34: // Interop.SizeOf(Type) on the GENERIC body: its generic branch reflection-invokes Unsafe.SizeOf<T>()
                case 36: // ... and on ArrayView1D<int, Dense> itself
                {
                    var t = probe == 34 ? typeof(ProbeGenericGridBody<int, Stride1D.Dense>) : typeof(ArrayView1D<int, Stride1D.Dense>);
                    int size = ILGPU.Interop.SizeOf(t);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] {probe}: Interop.SizeOf({t.Name}) = {size}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 35: // a DynamicMethod boxing the generic body (the WebGPU launcher's shape), built and RUN once
                {
                    var bt = typeof(ProbeGenericGridBody<int, Stride1D.Dense>);
                    var dm = new System.Reflection.Emit.DynamicMethod("probe35", typeof(object), new[] { bt }, typeof(Studio).Module);
                    var il = dm.GetILGenerator();
                    il.Emit(System.Reflection.Emit.OpCodes.Ldarg_0);
                    il.Emit(System.Reflection.Emit.OpCodes.Box, bt);
                    il.Emit(System.Reflection.Emit.OpCodes.Ret);
                    var boxed = dm.Invoke(null, new object[] { default(ProbeGenericGridBody<int, Stride1D.Dense>) });
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] 35: DynamicMethod boxed {boxed?.GetType().Name}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 37: // COMPILE ONLY (frontend IL -> IR -> WGSL, no launcher, no kernel object) the generic-body grid-stride kernel
                {
                    var mi = typeof(GridExtensions).GetMethod(nameof(GridExtensions.GridStrideLoopKernel))!
                        .MakeGenericMethod(typeof(ProbeGenericGridBody<int, Stride1D.Dense>));
                    var entry = ILGPU.Backends.EntryPoints.EntryPointDescription.FromExplicitlyGroupedKernel(mi);
                    var compiled = accel.CompileKernel(entry);
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] 37: compiled {compiled.GetType().Name}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 38: // DynamicMethod launcher shape with the GENERIC body by value, CreateDelegate (bound), never invoked
                case 39: // control: the same with the closed ProbeBody
                {
                    var bt = probe == 38 ? typeof(ProbeGenericGridBody<int, Stride1D.Dense>) : typeof(ProbeBody);
                    var dm = new System.Reflection.Emit.DynamicMethod("probe" + probe, typeof(void),
                        new[] { typeof(object), typeof(AcceleratorStream), typeof(LongIndex1D), bt }, typeof(Studio).Module);
                    var il = dm.GetILGenerator();
                    il.Emit(System.Reflection.Emit.OpCodes.Ldarg_3);
                    il.Emit(System.Reflection.Emit.OpCodes.Box, bt);
                    il.Emit(System.Reflection.Emit.OpCodes.Pop);
                    il.Emit(System.Reflection.Emit.OpCodes.Ret);
                    var delType = typeof(Action<,,>).MakeGenericType(typeof(AcceleratorStream), typeof(LongIndex1D), bt);
                    var del = dm.CreateDelegate(delType, new object());
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] {probe}: bound {del.GetType().Name} over {bt.Name}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 40: // a regular GENERIC C# method instantiated over the generic body (MakeGenericMethod), bound delegate
                {
                    var bt = typeof(ProbeGenericGridBody<int, Stride1D.Dense>);
                    var mi = typeof(Studio).GetMethod(nameof(ProbeLauncherSink), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!
                        .MakeGenericMethod(bt);
                    var delType = typeof(Action<,,>).MakeGenericType(typeof(AcceleratorStream), typeof(LongIndex1D), bt);
                    var del = mi.CreateDelegate(delType, new object());
                    using var b = accel.Allocate1D<int>(4095);
                    Console.WriteLine($"[AotProbe] 40: bound generic-method delegate over {bt.Name}, replica copy with HasNoData()");
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 42: // WARM first: the int copy paths run once BEFORE any poisoning delegate exists; then poison; then again
                {
                    using (var w = accel.Allocate1D(new int[16])) { }
                    using (var w2 = accel.Allocate1D<int>(16)) ReplicaCopyHasNoData(w2.View, accel.DefaultStream, new int[16]);
                    await accel.SynchronizeAsync();
                    var init = accel.CreateInitializer<int, Stride1D.Dense>();   // the poison (probe 22)
                    Console.WriteLine("[AotProbe] 42: warmed, poisoned, now the same int uploads again");
                    using var b = accel.Allocate1D(new int[4095]);
                    ReplicaCopyHasNoData(b.View, accel.DefaultStream, new int[4095]);
                    break;
                }
                case 7: // uint[4095]
                {
                    var a = new uint[4095];
                    using var b = accel.Allocate1D(a);
                    break;
                }
                default:
                    // A probe this build does not have must not read as a pass (a stale publish printed PASS 13-15).
                    Console.WriteLine($"[AotProbe] FAIL {probe}: no such probe in this build");
                    return;
            }
            await accel.SynchronizeAsync();
            Console.WriteLine($"[AotProbe] PASS {probe}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AotProbe] FAIL {probe}: {ex.GetType().Name}: {ex.Message}");
        }
    }
}
