using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;

namespace OpenMined.Network.Servers
{
    public class Request
    {
        public class EthResponse
        {
            public string jsonrpc;
            public int id;
        }

        public class blockNumber : EthResponse
        {
            public string result;
        }

        public static string infura_URL = "https://api.infura.io/v1/jsonrpc/";
        public static string infura_network = "rinkeby/";

        public Coroutine coroutine { get; private set; }
        public object result;
        private IEnumerator target;

        public Request(MonoBehaviour owner, IEnumerator target)
        {
            this.target = target;
            this.coroutine = owner.StartCoroutine(Run());
        }

        private IEnumerator Run()
        {
            while (target.MoveNext())
            {
                result = target.Current;
                yield return result;
            }
        }

        public static IEnumerator Get<T>(string method) where T : EthResponse
        {
            string URL = infura_URL + infura_network + method;

            Debug.LogFormat("Request.Get {0}", URL);
            UnityWebRequest www = UnityWebRequest.Get(URL);
            www.SetRequestHeader("accept", "application/json");

            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
                yield return null;
            }
            else
            {
                string json = www.downloadHandler.text;

                T response = JsonUtility.FromJson<T>(json);
                yield return response;
            }
        }

        public static IEnumerator GetBlockNumber(MonoBehaviour owner)
        {
            Request req = new Request(owner, Request.Get<Request.blockNumber>("eth_blockNumber"));
            yield return req.coroutine;

            Request.blockNumber response = req.result as Request.blockNumber;
            int result = (int)new System.ComponentModel.Int32Converter().ConvertFromString(response.result);
            Debug.LogFormat("\nCurrent Rinkeby Block Number: {0}", result.ToString("N"));
        }
    }
}
