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
        => Matrix4x4.CreateTranslation(-headPosition) * RoomToSceneYaw(headOrientation, cameraForward) * Matrix4x4.CreateTranslation(cameraPosition);

    /// <summary>
    /// AR start: the scene as a miniature in front of the viewer - its robust bounds (<see cref="SplatBounds.ComputeRobustAsync"/>)
    /// scaled so the largest side is <paramref name="size"/> metres, the bottom centre <paramref name="distance"/> metres
    /// ahead at about table height, turned so the viewer looks into it the way the desktop camera did. Full size around
    /// the viewer (the VR start) would cover most of the passthrough with a scene seen from inside.
    /// </summary>
    public static Matrix4x4 SceneFromRoomMiniature(Vector3 headPosition, Quaternion headOrientation, Vector3 cameraForward,
        SplatBounds.Aabb box, float size = 0.6f, float distance = 0.9f)
    {
        float largest = MathF.Max(box.MaxX - box.MinX, MathF.Max(box.MaxY - box.MinY, box.MaxZ - box.MinZ));
        float scale = MathF.Max(largest, 1e-6f) / size;   // scene units per room metre
        var anchorScene = new Vector3(box.CentreX, box.MinY, box.CentreZ);
        var f = Vector3.Transform(-Vector3.UnitZ, headOrientation);
        var flat = new Vector3(f.X, 0, f.Z);
        flat = flat.LengthSquared() < 1e-8f ? -Vector3.UnitZ : Vector3.Normalize(flat);
        var anchorRoom = headPosition + flat * distance;
        anchorRoom.Y = Math.Clamp(headPosition.Y - 0.6f, 0.3f, 1.0f);   // local-floor: about a table under a standing head
        return Matrix4x4.CreateTranslation(-anchorRoom) * Matrix4x4.CreateScale(scale)
            * RoomToSceneYaw(headOrientation, cameraForward) * Matrix4x4.CreateTranslation(anchorScene);
    }

    /// <summary>The turn about Y that carries the head's horizontal facing (room) onto the camera's (scene).</summary>
    static Matrix4x4 RoomToSceneYaw(Quaternion headOrientation, Vector3 cameraForward)
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
        return Vector3.Dot(Vector3.TransformNormal(hf, a), cf) >= Vector3.Dot(Vector3.TransformNormal(hf, b), cf) ? a : b;
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

    /// <summary>
    /// A room view matrix (room -> eye) as a RIGID scene view matrix (scene -> eye, in scene units). With the scene scaled
    /// by the grips (<see cref="XRWorldGrab"/>), room -> eye after scene -> room carries that scale, and the splat shader
    /// rebuilds a rigid camera (right/up/forward/position) from the view: the first scaled session drew a shrunken room of
    /// huge blobs (emulator, 2026-10-03). Scaling eye space about the eye changes no pixel (x/z and y/z keep their
    /// ratios), so the eye coordinates are put back in scene units; the scale still shows, through where each eye sits in
    /// the scene (the eye separation in scene units).
    /// </summary>
    public static Matrix4x4 SceneView(Matrix4x4 roomView, Matrix4x4 sceneFromRoom)
    {
        Matrix4x4.Invert(sceneFromRoom, out var roomFromScene);
        return roomFromScene * roomView * Matrix4x4.CreateScale(XRWorldGrab.Scale(sceneFromRoom));
    }
}
