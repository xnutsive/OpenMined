using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

namespace OpenMined.Network.Utils
{
    public static class EthereumAbiUtil
    {
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
            var hs = hexString.Substring(64 * parameter + 2, 64);

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
    }
}
