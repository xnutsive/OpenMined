using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Model
    {
        public readonly List<Model> Layers;

        public Model(params Model[] layers)
        {
            Layers = new List<Model>();
            
            // TODO -- dumb way of doing this
            foreach (var layer in layers)
            {
                Layers.Add(layer);
            }
        }

        public FloatTensor Predict(FloatTensor input)
        {
            
            foreach (var layer in Layers)
            {
                input = layer.Forward(input);
            }

            return input;
        }

        protected virtual FloatTensor Forward(FloatTensor input)
        {
            // Model layer must implement forward
            throw new NotImplementedException();
        }

        [CanBeNull]
        public virtual FloatTensor GetWeights()
        {
            return null;
        }
    }
}