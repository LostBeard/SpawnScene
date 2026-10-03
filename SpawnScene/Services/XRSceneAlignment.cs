using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Where an XR session's room sits in the scene. WebXR poses are in the room's local-floor space (the headset starts
/// near (0, 1.7, 0)); a scene is in its own coordinates, so drawing the room's views unchanged put the viewer wherever
/// the room origin happened to fall in the scene - the first SpawnScene VR session in the emulator showed nothing at all
/// (2026-10-03). This maps the room so the HEAD starts where the desktop viewer's camera is, facing the same horizontal
/// direction. Both spaces are Y-up (scenes are aligned to <see cref="ImageOrientation.WorldUp"/>), so the map is a turn
/// about Y plus a translation; scale stays 1 for now.
/// </summary>
public static class XRSceneAlignment
{
    /// <summary>
    /// Room-to-scene transform (row-vector convention: pScene = pRoom * result) that puts <paramref name="headPosition"/>
    /// at <paramref name="cameraPosition"/> and turns the head's horizontal facing onto the camera's.
    /// </summary>
    public static Matrix4x4 SceneFromRoom(Vector3 headPosition, Quaternion headOrientation, Vector3 cameraPosition, Vector3 cameraForward)
    {
        // WebXR views look down -Z.
        var headForward = Vector3.Transform(-Vector3.UnitZ, headOrientation);
        float Heading(Vector3 f) => MathF.Atan2(f.X, -f.Z);   // 0 = facing -Z, +pi/2 = facing +X
        bool flatHead = headForward.X * headForward.X + headForward.Z * headForward.Z < 1e-8f;
        bool flatCam = cameraForward.X * cameraForward.X + cameraForward.Z * cameraForward.Z < 1e-8f;
        float turn = flatHead || flatCam ? 0f : Heading(cameraForward) - Heading(headForward);
        // CreateRotationY's sign convention: pick the direction that carries the head's heading onto the camera's.
        var a = Matrix4x4.CreateRotationY(turn);
        var b = Matrix4x4.CreateRotationY(-turn);
        var hf = Vector3.Normalize(new Vector3(headForward.X, 0, headForward.Z) + new Vector3(1e-9f, 0, 0));
        var cf = Vector3.Normalize(new Vector3(cameraForward.X, 0, cameraForward.Z) + new Vector3(1e-9f, 0, 0));
        var rot = Vector3.Dot(Vector3.TransformNormal(hf, a), cf) >= Vector3.Dot(Vector3.TransformNormal(hf, b), cf) ? a : b;
        return Matrix4x4.CreateTranslation(-headPosition) * rot * Matrix4x4.CreateTranslation(cameraPosition);
    }

    /// <summary>
    /// A WebXR matrix (16 floats, column-major, for column vectors: v' = P v) as a System.Numerics matrix (row vectors:
    /// v' = v M, the renderer's convention - mvp = view * proj). M is P transposed, and P's column-major array read in
    /// order IS P transposed, so the floats go in row by row as they come. The transposed read used before handed the
    /// column-vector P to a row-vector pipeline: the perspective divide picked up a scaled depth term, so XR showed a
    /// zoomed-in slice with splats shrunk to dots (emulator, 2026-10-03).
    /// </summary>
    public static Matrix4x4 FromWebXRMatrix(float[] m) => new(
        m[0], m[1], m[2], m[3],
        m[4], m[5], m[6], m[7],
        m[8], m[9], m[10], m[11],
        m[12], m[13], m[14], m[15]);

    /// <summary>A room view matrix (room -> eye) as a scene view matrix (scene -> eye).</summary>
    public static Matrix4x4 SceneView(Matrix4x4 roomView, Matrix4x4 sceneFromRoom)
    {
        Matrix4x4.Invert(sceneFromRoom, out var roomFromScene);
        return roomFromScene * roomView;
    }
}
