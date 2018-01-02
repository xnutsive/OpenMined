using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;

namespace OpenMined.Syft.Optim
{
    public class SGD
    {
        private SyftController ctrl;
        private List<int> parameters;
        private float alpha;
        
        // Should we put a check incase this variable overflows?
        protected static volatile int nCreated = 0;
        protected int id;
        
        public int Id
        {
            get { return id; }
            protected set { id = value; }
        }

        public SGD(SyftController ctrl_, List<int> parameters_, float alpha_)
        {
            this.ctrl = ctrl_;
            this.parameters = parameters_;
            this.alpha = alpha_;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            ctrl.addOptimizer(this);
        }

        public void ZeroGrad()
        {
            foreach (int param_index in parameters)
                if(ctrl.floatTensorFactory.Get(param_index).Grad != null)
                    ctrl.floatTensorFactory.Get(param_index).Grad.Zero_();
        }

        public void Step(int batch_size)
        {
            foreach (int param_index in parameters)
            {
                var param = ctrl.floatTensorFactory.Get(param_index);
                param.Sub(param.Grad.Mul(alpha/(float)batch_size),inline:true);
            }
        }
        
        public string ProcessMessage (Command msgObj, SyftController ctrl)
        {

            switch (msgObj.functionCall)
            {
                case "zero_grad":
                    ZeroGrad();
                    return "";
                case "step":
                    Step(int.Parse(msgObj.tensorIndexParams[0]));
                    return "";
               
            }

            throw new InvalidOperationException("Could not find function for command:" + msgObj.functionCall);

        }
    }
}