using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Policy:Layer
    {
        private Layer model;
        
        public Policy(SyftController _controller, Layer _model)
        {
            init("policy");
            controller = _controller;
            
            model = _model;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);

        }

        public override FloatTensor Forward(FloatTensor input)
        {
            return model.Forward(input);
        }

        /*public int[] Sample(FloatTensor input)
        {
            
        }*/
    }
}