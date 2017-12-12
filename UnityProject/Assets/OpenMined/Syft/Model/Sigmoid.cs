using System;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Sigmoid: Model
    {
        protected override FloatTensor Forward(FloatTensor input)
        {
            return input.Sigmoid();
        }
    }
}