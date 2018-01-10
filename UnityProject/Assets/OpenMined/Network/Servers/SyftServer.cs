using UnityEngine;
using OpenMined.Network.Controllers;
using UnityEngine.Networking;
using System.Collections;
using OpenMined.Syft.Tensor;
using OpenMined.Syft.Tensor.Factories;

namespace OpenMined.Network.Servers
{

	public class SyftServer : MonoBehaviour
	{
		public bool Connected;
		private NetMqPublisher _netMqPublisher;
		private string _response;

		public SyftController controller;

		[SerializeField] private ComputeShader shader;

		private IEnumerator Start()
		{
			_netMqPublisher = new NetMqPublisher(HandleMessage);
			_netMqPublisher.Start();

			controller = new SyftController(shader);

            yield return Request.GetBlockNumber(this);

            yield return Request.GetNumModels(this);

            yield return Request.GetModel(this, 13);

            // yield return Ipfs.WriteIpfs();

            var tensor = Ipfs.Get("QmWi4Y2qyBTuztP3RP7AgEMX9p2mb4VsX1mS3EPvTedvZV");
            if (tensor != null) 
            {
                Debug.Log("Got the thing: " + tensor);
            }

            IpfsModel model = Ipfs.GetModel("QmRDrHMEd7F4Ueoh5pWm97FZnQorjkneyHcEk9NcBPQZg9");
            if (model != null)
            {
                Debug.Log("Got the IpfsModel: " + model);
            }

		}

		private void Update()
		{
			_netMqPublisher.Update();
		}

		private string HandleMessage(string message)
		{
			//Debug.LogFormat("HandleMessage... {0}", message);
			return controller.processMessage(message, this);
		}

		private void OnDestroy()
		{
			_netMqPublisher.Stop();
		}

		public ComputeShader Shader {
			get { return shader; }
		}
	}
}
