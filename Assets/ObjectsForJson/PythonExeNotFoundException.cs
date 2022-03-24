using System;


namespace ObjectsForJson
{
    [Serializable]
    public class PythonExeNotFoundException : Exception
    {
        public PythonExeNotFoundException() : base("Python Executable file not found.") { }
        public PythonExeNotFoundException(string message) : base(message) { }
        public PythonExeNotFoundException(string message, Exception inner) : base(message, inner) { }

        // A constructor is needed for serialization when an
        // exception propagates from a remoting server to the client.
        protected PythonExeNotFoundException(System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}