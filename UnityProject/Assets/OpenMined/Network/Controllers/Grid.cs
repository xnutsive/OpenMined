using UnityEngine;
using System;
using System.Collections.Generic;
using System.Collections;
using OpenMined.Network.Utils;
using OpenMined.Network.Servers;
using OpenMined.Syft.Tensor;
using OpenMined.Syft.Layer;

namespace OpenMined.Network.Controllers
{
    public class Grid
    {

        private SyftController controller;

        public Grid(SyftController controller)
        {
            this.controller = controller;
        }

        public void Run(int inputId, int targetId, List<GridConfiguration> configurations)
        {
            var inputTensor = new IpfsTensor(controller.floatTensorFactory.Get(inputId));
            var targetTensor = new IpfsTensor(controller.floatTensorFactory.Get(targetId));

            // write the input and target tensors to Ipfs
            var inputJob = new Ipfs();
            var targetJob = new Ipfs();

            var inputIpfsResponse = inputJob.Write(inputTensor);
            var targetIpfsResponse = targetJob.Write(targetTensor);

            Debug.Log("Input Hash: " + inputIpfsResponse.Hash);
            Debug.Log("Target Hash: " + targetIpfsResponse.Hash);

            configurations.ForEach((config) => {
                var model = controller.getModel(config.model) as Sequential;
                var layers = model.getLayers();

                var serializedModel = new List<String>();

                layers.ForEach((layerId) => {
                    var layer = controller.getModel(layerId);
                    var namedLayer = layer as LayerDefinition;
                    if (namedLayer == null) return;

                    serializedModel.Add(namedLayer.GetLayerDefinition());
                });

                var configJob = new Ipfs();
                var response = configJob.Write(new IpfsModel(serializedModel, config.lr));

                Debug.Log("Model: " + response.Hash);
            });
        }
    }

    public interface LayerDefinition {
        string GetLayerDefinition();
    }

    [Serializable]
    public class IpfsModel {
        [SerializeField] List<String> Model;
        [SerializeField] float lr;

        public IpfsModel (List<String> model, float lr)
        {
            this.Model = model;
            this.lr = lr;
        }
    }
}
