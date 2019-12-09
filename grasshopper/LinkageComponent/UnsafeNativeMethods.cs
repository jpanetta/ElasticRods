using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace LinkageComponent
{
    internal static class Import
    {
        public const string lib = "libgrasshopper_bindings.dylib";
    }

    /// <summary>
    /// http://msdn.microsoft.com/en-us/library/aa288468(VS.71).aspx
    /// http://www.mono-project.com/docs/advanced/pinvoke/
    /// </summary>
    internal static class UnsafeNativeMethods
    {
        [DllImport(Import.lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rod_linkage_grasshopper_interface(int numVertices, int numEdge,
                double[] inCoords, int[] inEdges, double[] outCoordsClosed, double[] outCoordsDeployed,
                double angle, int openingSteps);
    }
}
