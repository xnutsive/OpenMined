using UnityEngine;
using OpenMined.Network.Controllers;
using UnityEngine.Networking;
using System.Collections;

namespace OpenMined.Network.Servers
{
    public class res_eth_blockNumber
    {
        public string jsonrpc;
        public int id;
        public string result;
    }

    public class SyftServer : MonoBehaviour
    {
        public bool Connected;
        private NetMqPublisher _netMqPublisher;
        private string _response;

        private SyftController controller;

        [SerializeField] private ComputeShader shader;

        public string infura_URL = "https://api.infura.io/v1/jsonrpc/";
        public string infura_network = "rinkeby/";

        IEnumerator Get(string method)
        {
            string URL = infura_URL + infura_network + method;

            Debug.LogFormat("http get {0}", URL);
            UnityWebRequest www = UnityWebRequest.Get(URL);
            www.SetRequestHeader("accept", "application/json");
            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
            }
            else
            {
                string json = www.downloadHandler.text;
                Debug.LogFormat("response json {0}", json);
                Debug.LogFormat("header content-type: {0}", www.GetResponseHeader("content-type"));
                Debug.LogFormat("header content-length: {0}", www.GetResponseHeader("content-length"));
                Debug.LogFormat("header server: {0}", www.GetResponseHeader("server"));

                res_eth_blockNumber response = JsonUtility.FromJson<res_eth_blockNumber>(json);
                Debug.LogFormat("parsed id: {0}", response.id);
                Debug.LogFormat("parsed hex result: {0}", response.result);

                int result = (int)new System.ComponentModel.Int32Converter().ConvertFromString(response.result);
                Debug.LogFormat("\nCurrent Rinkeby Block Number: {0}", result.ToString("N"));
            }
        }

        private void Start()
        {
            _netMqPublisher = new NetMqPublisher(HandleMessage);
            _netMqPublisher.Start();

            controller = new SyftController(shader);

            StartCoroutine(Get("eth_blockNumber"));
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

        public ComputeShader Shader
        {
            get { return shader; }
        }
    }
}
