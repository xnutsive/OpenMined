using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System.Collections.Generic;
using System;
using OpenMined.Network.Utils;

namespace OpenMined.Network.Servers
{
    public class Request
    {
        public class EthResponse
        {
            public string jsonrpc;
            public int id;
        }

        public class BlockNumber : EthResponse
        {
            public string result;
        }

        public class Call : EthResponse
        {
            public string result;
        }
        
        public class GetModelResponse
        {
            public String address = "";
            public Int32 bounty = 0;
            public Int32 initialError = 0;
            public Int32 targetError = 0;
            public String inputAddress = "";
            public String targetAddress = "";

            private int numParameters = 6;

            private readonly List<System.Type> types;

            public GetModelResponse(string hexString)
            {
                types = new List<System.Type>
                {
                    address.GetType(),
                    bounty.GetType(),
                    initialError.GetType(),
                    targetError.GetType(),
                    inputAddress.GetType(),
                    targetError.GetType()
                };

                var objects = EthereumAbiUtil.GetParametersHex(hexString, numParameters, types);

                address = (String)objects[0];
                bounty = (Int32)objects[1];
                initialError = (Int32)objects[2];
                targetError = (Int32)objects[3];
            }
        }

        public static string identityURL = "http://localhost:3000/";

        public static string contractAddress = "0xd60e1a150b59a89a8e6e6ff2c03ffb6cb4096205";
        public static string infuraURL = "https://api.infura.io/v1/jsonrpc/";
        public static string infuraNetwork = "rinkeby/";

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

        public static IEnumerator GetIdentity(string method, 
                                              string inputAddress = "", 
                                              string targetAddress = "")                                
        {
            string URL = identityURL;
            
            if(method.Length > 0)
            {
                var input = WWW.EscapeURL(inputAddress);
                var target = WWW.EscapeURL(targetAddress);
                URL += "/" + method + "?input=" + input + "&target=" + target;
            } 

            Debug.LogFormat("Request.GetIdentity {0}", URL);
            UnityWebRequest www = UnityWebRequest.Get(URL);
            www.SetRequestHeader("accept", "text/plain");

            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
                yield return null;
            }
            else
            {
                yield return www.downloadHandler.text;
            }
        }

        public static IEnumerator Get<T>(string method, string data = "") where T : EthResponse
        {
            string URL = infuraURL + infuraNetwork + method;

            if (data != "") {
                URL += "?params=" + data;
            }

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
            Request req = new Request(owner, Request.Get<Request.BlockNumber>("eth_blockNumber"));
            yield return req.coroutine;

            Request.BlockNumber response = req.result as Request.BlockNumber;
            int result = (int) new System.ComponentModel.Int32Converter().ConvertFromString(response.result);
            Debug.LogFormat("\nCurrent Rinkeby Block Number: {0}", result.ToString("N"));
        }

        private static string encodeData(string data)
        {
            string encodedData = WWW.EscapeURL("[{\"to\":\"" + contractAddress + "\",\"data\":\"" + data + "\"},\"latest\"]");
            return encodedData;
        }

        public static IEnumerator GetNumModels(MonoBehaviour owner)
        {
            // TODO: convert "getNumModels" to hex.
            string data = encodeData("0x3c320cc2");

            Request req = new Request(owner, Request.Get<Request.Call>("eth_call", data));
            yield return req.coroutine;

            Request.Call response = req.result as Request.Call;
            int result = (int)new System.ComponentModel.Int32Converter().ConvertFromString(response.result);
            Debug.LogFormat("\nNum Models: {0}", result.ToString("N"));
        }

        public static IEnumerator GetModel(MonoBehaviour owner, int modelId)
        {
            var keccak = new Sha3Keccak();

            var d = keccak.CalculateHash("getModel(uint256)");

            Debug.LogFormat("keccak {0}", d);
            
            // TODO: convert "getModel" and modelId to hex.
            string data = encodeData("0x6d3616940000000000000000000000000000000000000000000000000000000000000001");

            Request req = new Request(owner, Request.Get<Request.Call>("eth_call", data));
            yield return req.coroutine;

            Request.Call response = req.result as Request.Call;
            Debug.Log(response.result);

            var modelResponse = new GetModelResponse(response.result);

            Debug.LogFormat("Model {0}, {1}. {2}, {3}, {4}, {5}", modelResponse.address, modelResponse.bounty, modelResponse.initialError, modelResponse.targetError, modelResponse.inputAddress, modelResponse.targetAddress);
        }
        
        
        public static IEnumerator AddModel(MonoBehaviour owner)
        {
            Request req = new Request(owner, Request.GetIdentity("addModel", "bleh", "bleh2"));
            yield return req.coroutine;

            Debug.LogFormat("response {0}", req.result);
        }
    }
}
