# SpawnScene.Tests

CPU geometry gates — **no browser, no GPU, no manual Studio clicking**.

```powershell
cd D:\users\tj\Projects\SpawnScene
dotnet test SpawnScene.Tests -c Release
```

Filter:

```powershell
dotnet test SpawnScene.Tests -c Release --filter TempleRing
```

`TempleRingWorldSpaceTests` proves Middlebury unproject / multi-view fusion math. If these are green and Studio still looks wrong, the bug is depth↔pose coupling or the deployed build — not the camera convention.
