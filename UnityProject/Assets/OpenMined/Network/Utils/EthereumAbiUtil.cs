using UnityEngine;
using System;
using System.Collections.Generic;
using OpenMined.Network.Servers;
using OpenMined.Hex.HexConvertors.Extensions;

namespace OpenMined.Network.Utils
{
    public static class EthereumAbiUtil
    {
        // TODO if we keep on going down this route of implementing the ethereum JSON-RPC 
        // by ourseleves we probably want to store hex vals as byte arrays
        // TODO only really supports decoding hex strings (byte arrays??) and ints
        // and encoding ints
        public static string contractAddress = "0xd60e1a150b59a89a8e6e6ff2c03ffb6cb4096205";

        public static object[] GetParametersHex(string hexString, int parameters, List<System.Type> types)
        {
            var objects = new object[6];
            for (int i = 0; i < parameters; i++)
            {
                objects[i] = EthereumAbiUtil.GetParameter(hexString, i, types[i]);
            }

            return objects;
        }
        
        public static object GetParameter(string hexString, int parameter, System.Type type)
        {
            var hs = hexString.RemoveHexPrefix();
            hs = hs.Substring(64 * parameter, 64);

            hs = EthereumAbiUtil.StripPadding(hs);

            if (type.Name == "String")
            {
                return (String)hs;
            }
            else if (type.Name == "Int32")
            {
                return EthereumAbiUtil.ConvertToInt(hs);
            }

            return hs;
        }

        public static Int32 ConvertToInt(string hexString)
        {
            Int32 decval = System.Convert.ToInt32(hexString, 16);

            return decval;
        }

        public static string StripPadding(string hexString)
        {
            var index = 0;
            for (int j = 0; j < hexString.Length; j++)
            {
                if (hexString[j] != '0')
                {
                    index = j;
                    break;
                }
            }
                        
            return hexString.Remove(0, index);
        }

        public static string EncodeEthereumFunctionCall(string funcSig, object[] parameters)
        {
            var keccak = new Sha3Keccak();

            var hash = keccak.CalculateHash(funcSig);

            var encodedFunc = EthereumAbiUtil.FirstFourBytesKeccak(hash);

            foreach (var parameter in parameters)
            {
                if (parameter.GetType().Name == "Int32")
                {
                    encodedFunc += EncodeInt32((Int32)parameter);
                }  
            }

            return EthereumAbiUtil.EncodeAsJSON("0x" + encodedFunc);
        }

        public static string EncodeInt32(Int32 parameter)
        {
            return PadDataTo32Bytes(parameter.ToString("X"));
        }
        
        private static string EncodeAsJSON(string data)
        {
            string encodedData = WWW.EscapeURL("[{\"to\":\"" + contractAddress + "\",\"data\":\"" + data + "\"},\"latest\"]");
            return encodedData;
        }

        public static string FirstFourBytesKeccak(string hexString)
        {
            return hexString.Substring(0, 8);
        }

        public static string PadDataTo32Bytes(string hexString)
        {
            if (hexString.Length == 64)
            {
                return hexString;
            }
            
            var padLength = 64 - hexString.Length;

            var pad = hexString.PadLeft(padLength, '0');

            return pad;
        }
    }
}
