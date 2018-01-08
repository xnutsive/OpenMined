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

            yield return Request.GetModel(this, 1);

            float[] data = { 0.5f, 1f, 2f, 3f, 0.434343f };
            int[] shape = { 5 };
            var t = controller.floatTensorFactory.Create(_shape: shape, _data: data, _shader: controller.Shader);
            var ipfsReq = new IpfsTensor(t);
            yield return Ipfs.WriteIpfs(ipfsReq);

            // yield return Ipfs.WriteIpfs();

            // yield return Ipfs.GetIpfs();
		}

		private void Update()
		{
			_netMqPublisher.Update();
		}

		private string HandleMessage(string message)
		{
			//Debug.LogFormat("HandleMessage... {0}", message);
			return controller.processMessage(message);
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
