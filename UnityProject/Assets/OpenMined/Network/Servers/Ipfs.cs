using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System;


namespace OpenMined.Network.Servers
{
    public static class Ipfs
    {

        public static string POST_URL = "https://ipfs.infura.io:5001/api/v0/add?stream-channels=true";
        public static string GET_URL = "https://ipfs.infura.io/ipfs";

        public static IEnumerator WriteIpfs<T>(T data)
        {
            var serializedData = JsonUtility.ToJson(data);
            var stringData = "--------------------------30a67cb5e62650e3\r\nContent-Disposition: form-data; name=\"file\"; filename=\"model\";\r\n";
            stringData    += "Content-Type: application/octet-stream\r\n\r\n";
            stringData    += serializedData + "\r\n";
            stringData    += "--------------------------30a67cb5e62650e3--\r\n";

            var bytes = System.Text.Encoding.UTF8.GetBytes(stringData);
            UnityWebRequest www = UnityWebRequest.Put(Ipfs.POST_URL, bytes);
            www.SetRequestHeader("Content-Type", "multipart/form-data; boundary=------------------------30a67cb5e62650e3");
            yield return www.SendWebRequest();
            if (www.isHttpError || www.isNetworkError)
            {
                Debug.Log("Error making IPFS request: " + www.error);
                yield return null;
            }
            else 
            {
                string json = www.downloadHandler.text;
                IpfsResponse response = JsonUtility.FromJson<IpfsResponse>(json);
                Debug.Log("Got Ipfs response: " + response);
                yield return response;
            }
        }

        public static IEnumerator GetIpfs (string path)
        {
            var www = UnityWebRequest.Get(GET_URL + "/" + path);
            yield return www.SendWebRequest();
            if (www.isHttpError || www.isNetworkError)
            {
                Debug.Log("Error getting IPFS data: " + www.error);
                yield return null;
            }
            else
            {
                var json = www.downloadHandler.text;
                Debug.Log("Got Ipfs response: " + json);

                yield return json;
            }
        }
    }

    [Serializable]
    public class IpfsResponse
    {
        public string Name;
        public string Hash;
        public string Size;
    }
}
