NOTE: This is *not* the real Rhino plugin; it is just a very basic interface
for interactively running the deployment code.

Using the Grasshopper Module on Mac:

1. Pull the latest version from elastic_rods git repo
2. build 
3. Install Visual Studio from
https://visualstudio.microsoft.com/vs/mac/
4. Install Grasshopper Plugin for Visaul Studio
https://github.com/mcneel/RhinoCommonXamarinStudioAddin/releases 
Rhino mpack
In Visaul Studio go to the Visual Studio Community Toolbar -> Extensions. -> Install from File button -> select the Rhino....mpack file
5. Restart Visual Studio
6. Open .sln project from elastic_rods/grasshopper/LinkageComponent/ in Visual studio and Run it. This will open Rhino and you should also open Grasshopper. In Grasshopper open XshellsInterface.gh from elastic_rods/grasshopper/.
7. After running the .sln project it will produce the .dylib and .gha files in elastic_rods/grasshopper/LinkageComponent/bin/Release/. To make the Linkage component visible to Grasshopper every time you run it, copy this to files in the Grasshopper component library (Open Grasshopper, in Grasshopper open Files -> Special Folders -> Components  Folder).
