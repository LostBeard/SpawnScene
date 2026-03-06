# Goals and Features
- Uses SpawnDev.BlazorJS, SpawnDev.ILGPU, and SpawnDev.BlazorJS.OnnxRuntimeWeb to build a Blazor Wasm app that can use Guassian Splatting to create 3D scenes from images wit hthe help of monocular depth estimation.
- The goals are to showcase the capabilities of the various SpawnDev libraries and Blazor WebAssembly in a practical application while demonstrating new and amazing techniques in 3D world mapping.



# Export to other formats
- 3D printer formats
- CAD formats
- 360 degree video formats
- PLY Export: For high-fidelity visual viewing (Gaussian Splatting).
- STL/OBJ Export: A "Solidify" button that triggers your ILGPU Marching Cubes/Poisson kernel to generate a 3D-printable mesh from the SfM data.


## Features
- View in VR/AR: Integrate WebXR to allow users to view their generated 3D scenes in virtual or augmented reality.
- Object isolation: Implement a feature that allows users to isolate and export specific objects from the generated 3D scene, which could be useful for various applications such as 3D printing or game development.
- 

Sugar (Surface-Aligned Gaussian Splatting): 
This is a specific method designed to bind gaussians to a mesh surface during the training process. 
It makes extracting a high-quality 3D mesh much more viable than standard 3DGS.


# Blazor Wasm Gaussian Splatting
This project is a Blazor WebAssembly application that demonstrates the use of Gaussian Splatting to create 3D scenes from images. It utilizes the SpawnDev.BlazorJS and SpawnDev.ILGPU libraries to achieve this functionality.

## Features
- **Gaussian Splatting**: Create 3D scenes from 2D images using Gaussian Splatting techniques.
- **Blazor WebAssembly**: A client-side web application framework that allows for rich interactive experiences in the browser.
- **SpawnDev Libraries**: Leverages the capabilities of SpawnDev.BlazorJS for JavaScript interop and SpawnDev.ILGPU for GPU computing.
- **Interactive 3D Scenes**: Users can interact with the generated 3D scenes, exploring the results of the Gaussian Splatting process.
- **Educational Purpose**: This project serves as a practical example of how to use the SpawnDev libraries in a real-world application, showcasing their potential in the field of 3D graphics and web development.
- **Open Source**: The project is open source, allowing developers to contribute and learn from the codebase.
- View generated 3D scenes using WebXR on compatible devices, providing an immersive experience.
- Support for various image formats to create 3D scenes, enhancing the versatility of the application.
- Performance optimizations using ILGPU to ensure smooth rendering of complex scenes in the browser.
- Supports VR headsets for an immersive experience when viewing the generated 3D scenes, allowing users to explore the results in a virtual reality environment.
- Create large and complex 3D scenes from high-resolution images, demonstrating the power of Gaussian Splatting in handling detailed visual data.
- Integration with WebXR to enable users to view and interact with the generated 3D scenes in virtual reality, providing an immersive experience that enhances the visualization of the results.
- Support for real-time updates to the 3D scenes, allowing users to see changes as they adjust parameters or upload new images, making the application more interactive and engaging.
- Cross-platform compatibility, ensuring that users can access the application from various devices and browsers without any issues, making it widely accessible to a broad audience interested in 3D graphics and web development.
- Scan and process images in real-time using the device's (VR headset, cell phone, tablet, PC, etc) camera, allowing users to create 3D scenes on the fly and explore their surroundings in a new way, enhancing the interactivity and utility of the application.
- Peer-to-peer using WebRTC to allow multi-device cooperative scene scanning and generation, enabling users to collaborate in creating and exploring 3D scenes together, fostering a sense of community and shared creativity within the application.
- Esaily share your saved 3D scenes with others by generating shareable links or exporting them in common 3D file formats, making it easy for users to showcase their creations and collaborate with others in the community.
- Integration with social media platforms to allow users to share their generated 3D scenes directly from the application, increasing visibility and engagement with the content created by users, and fostering a sense of community around the application.
- Support for custom shaders and materials, allowing users to further customize the appearance of their 3D scenes and create unique visual effects, enhancing the creative possibilities within the application.
- Integration with machine learning models to enhance the Gaussian Splatting process, allowing for improved scene generation and more accurate representations of the original images, pushing the boundaries of what can be achieved with this technique in a web-based application.
- Support for collaborative editing of 3D scenes, allowing multiple users to work on the same scene simultaneously, fostering a sense of community and shared creativity within the application, and enabling users to learn from each other and create more complex and interesting scenes together.
- Generate 3D scenes from video input, allowing users to create dynamic and evolving scenes that change over time, providing a new way to visualize and interact with video content in a 3D space.
- 