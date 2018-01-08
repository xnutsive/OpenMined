using UnityEngine;
using System;
using System.Collections.Generic;
using System.Collections;
using OpenMined.Network.Utils;
using OpenMined.Network.Servers;

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
        }
    }
}
