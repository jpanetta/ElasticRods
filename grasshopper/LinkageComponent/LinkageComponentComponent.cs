using System;
using System.Collections.Generic;
using System.IO;

using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;

namespace LinkageComponent
{
    public class LinkageComponentComponent : GH_Component
    {
        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public LinkageComponentComponent()
          : base("LinkageComponent", "RodLinkage",
            "Run deployment simulation.",
            "XShells", "Linkage Simulation")
        {
            GH_AssemblyInfo info = Instances.ComponentServer.FindAssembly(new Guid("b8a730a7-bdde-4d9b-b2b9-ea1b8e126bc2"));
            var path = Path.GetDirectoryName(info.Location);

            // Setting the path to the current component directory, therefore `.dylib`s put on the same folder will be discovered.
            Directory.SetCurrentDirectory(path);
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            // Use the pManager object to register your input parameters.
            pManager.AddNumberParameter("Vertices",      "V",     "List of vertex coordinates",      GH_ParamAccess.list);
            pManager.AddIntegerParameter("Edges",        "E",     "List of edge ids",                GH_ParamAccess.list);
            pManager.AddNumberParameter("Angle",         "Angle", "Opening angle",                   GH_ParamAccess.item,0.0);
            pManager.AddIntegerParameter("OpeningSteps", "Steps", "Number of steps opening linkage", GH_ParamAccess.item, 15);
            pManager.AddBooleanParameter("Solver ON", "Solver ON", "Turn on solver if true", GH_ParamAccess.item, true);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            //pManager.AddNumberParameter("Closed Vertices", "Closed V", "List of closed vertex coordinates", GH_ParamAccess.list); 
            //pManager.AddNumberParameter("Deployed Vertices", "Deployed V", "List of deployed vertex coordinates", GH_ParamAccess.list);
            // --- Grasshopper always reports an error if trying to output a list, hence we export an item of type Matrix 
            pManager.AddMatrixParameter("Closed Vertices", "Closed V", "List of closed vertex coordinates", GH_ParamAccess.item);
            pManager.AddMatrixParameter("Deployed Vertices", "Deployed V", "List of deployed vertex coordinates", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<double> vertices = new List<double>();
            List<int> edges = new List<int>();
            double angle = 0.0;
            int openingSteps = 0;
            bool solverOn = true;

            if (!DA.GetDataList(0, vertices)) return;
            if (!DA.GetDataList(1, edges)) return;
            if (!DA.GetData(2, ref angle)) return;
            if (!DA.GetData(3, ref openingSteps)) return;
            if (!DA.GetData(4, ref solverOn)) return;

            angle *= (Math.PI / 180.0); // conversion from degrees to radians 
            int numV = vertices.Count / 3;
            int numE = edges.Count / 2;

            Matrix verticesClosed = new Matrix(3, numV);
            Matrix verticesDeployed = new Matrix(3, numV);

            if (solverOn)
            {
                double[] v = new double[3*numV];
                //v = vertices.ToArray();
                for (var i = 0; i < (3 * numV); i++)
                    v[i] = vertices[i];

                int[] e = new int[2*numE];
                for (var i = 0; i < (2 * numE); i++)
                    e[i] = edges[i];

                double[] verticesC = new double[3 * numV];
                double[] verticesD = new double[3 * numV];

                UnsafeNativeMethods.rod_linkage_grasshopper_interface(numV, numE, v, e, verticesC, verticesD, angle, openingSteps);

                for (var i = 0; i < numV; i++)
                {
                    verticesClosed[0, i] = verticesC[3 * i];
                    verticesClosed[1, i] = verticesC[3 * i + 1];
                    verticesClosed[2, i] = verticesC[3 * i + 2];

                    verticesDeployed[0, i] = verticesD[3 * i];
                    verticesDeployed[1, i] = verticesD[3 * i + 1];
                    verticesDeployed[2, i] = verticesD[3 * i + 2];
                }
            }
            else 
            {
                for (var i = 0; i < numV; i++)
                {
                    verticesClosed[0, i] = vertices[3 * i];
                    verticesClosed[1, i] = vertices[3 * i + 1];
                    verticesClosed[2, i] = vertices[3 * i + 2];

                    verticesDeployed[0, i] = vertices[3 * i];
                    verticesDeployed[1, i] = vertices[3 * i + 1];
                    verticesDeployed[2, i] = vertices[3 * i + 2];
                }
            }

            DA.SetData(0, verticesClosed);
            DA.SetData(1, verticesDeployed);
        }

        /// <summary>
        /// The Exposure property controls where in the panel a component icon 
        /// will appear. There are seven possible locations (primary to septenary), 
        /// each of which can be combined with the GH_Exposure.obscure flag, which 
        /// ensures the component will only be visible on panel dropdowns.
        /// </summary>
        public override GH_Exposure Exposure
        {
            get { return GH_Exposure.primary; }
        }

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                // You can add image files to your project resources and access them like this:
                //return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("79563a5a-4a71-4691-9346-c4495d7d8107"); }
        }
    }
}
