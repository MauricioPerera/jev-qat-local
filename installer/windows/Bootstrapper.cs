// Single-file, per-user Windows launcher. Built from the committed source ZIP.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

[assembly: AssemblyTitle("Jev QAT Local - Instalador E2B")]
[assembly: AssemblyProduct("Jev QAT Local")]
[assembly: AssemblyCompany("MauricioPerera")]
[assembly: AssemblyDescription("Instalador de E2B QAT Q4 y su componente multimodal")]
[assembly: AssemblyVersion("@@VERSION@@.0")]
[assembly: AssemblyFileVersion("@@VERSION@@.0")]

internal static class Bootstrapper
{
    private const string Version = "@@VERSION@@";
    private const string PayloadHash = "@@PAYLOAD_SHA256@@";
    private const string Resource = "JevQat.Payload.zip";
    private static readonly string[] Files = {
        "Instalar E2B.cmd", "Instalar.ps1", "install.py", "launch.py",
        "source.zip", "bundle.json", "LEEME.txt", "LICENSE", "NOTICE.md"
    };
    private static readonly object OutputLock = new object();
    private static StreamWriter Log;
    private static volatile bool Cancelled;
    [DllImport("kernel32.dll")] private static extern IntPtr GetStdHandle(int kind);
    [DllImport("kernel32.dll")] private static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);

    private static void Say(string line)
    {
        lock (OutputLock)
        {
            Console.WriteLine(line);
            if (Log != null) { Log.WriteLine(line); Log.Flush(); }
        }
    }

    // CRT-compatible quoting; no cmd.exe or shell interpolation is involved.
    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char ch in value)
        {
            if (ch == '\\') { slashes++; continue; }
            if (ch == '"') { result.Append('\\', slashes * 2 + 1); }
            else { result.Append('\\', slashes); }
            result.Append(ch);
            slashes = 0;
        }
        result.Append('\\', slashes * 2);
        return result.Append('"').ToString();
    }

    private static byte[] Payload()
    {
        using (Stream resource = Assembly.GetExecutingAssembly().GetManifestResourceStream(Resource))
        {
            if (resource == null || resource.Length > 32 * 1024 * 1024)
                throw new InvalidDataException("Paquete interno no valido.");
            using (MemoryStream memory = new MemoryStream())
            {
                resource.CopyTo(memory);
                byte[] data = memory.ToArray();
                using (SHA256 sha = SHA256.Create())
                {
                    string actual = BitConverter.ToString(sha.ComputeHash(data)).Replace("-", "").ToLowerInvariant();
                    if (actual != PayloadHash) throw new InvalidDataException("SHA-256 del paquete interno incorrecto.");
                }
                return data;
            }
        }
    }

    private static void Validate(ZipArchive zip)
    {
        HashSet<string> expected = new HashSet<string>(Files, StringComparer.Ordinal);
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            if (!expected.Remove(entry.FullName) || entry.Length > 32 * 1024 * 1024)
                throw new InvalidDataException("Archivo interno inesperado: " + entry.FullName);
        }
        if (expected.Count != 0) throw new InvalidDataException("El paquete interno esta incompleto.");
    }

    private static string Extract(byte[] data)
    {
        using (ZipArchive zip = new ZipArchive(new MemoryStream(data), ZipArchiveMode.Read))
        {
            Validate(zip);
            string root = Path.Combine(Path.GetTempPath(), "JevQatLocal-Setup");
            string destination = Path.Combine(root, Version + "-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(destination);
            foreach (ZipArchiveEntry entry in zip.Entries)
            {
                // Validated names are a fixed list of flat files; never directories.
                using (Stream source = entry.Open())
                using (FileStream target = new FileStream(Path.Combine(destination, entry.FullName), FileMode.CreateNew))
                    source.CopyTo(target);
            }
            return destination;
        }
    }

    private static void Cleanup(string folder)
    {
        if (folder == null) return;
        try
        {
            string root = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "JevQatLocal-Setup"));
            string target = Path.GetFullPath(folder);
            if (!String.Equals(Path.GetDirectoryName(target), root, StringComparison.OrdinalIgnoreCase)
                || !Path.GetFileName(target).StartsWith(Version + "-", StringComparison.Ordinal)) return;
            // Only remove this launcher's known temporary files. No recursive deletion.
            foreach (string name in Files) File.Delete(Path.Combine(target, name));
            File.Delete(Path.Combine(target, "start.gate"));
            Directory.Delete(target, false);
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static int Run(string folder, List<string> options)
    {
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = Path.Combine(Environment.SystemDirectory, @"WindowsPowerShell\v1.0\powershell.exe");
        start.Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File " + Quote(Path.Combine(folder, "Instalar.ps1"));
        foreach (string option in options) start.Arguments += " " + Quote(option);
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        start.RedirectStandardInput = true;
        start.StandardOutputEncoding = new UTF8Encoding(false);
        start.StandardErrorEncoding = new UTF8Encoding(false);
        start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        // A parent PowerShell 7 may supply Core-only modules. This child always
        // uses Windows PowerShell 5.1 and must discover its own built-in modules.
        start.EnvironmentVariables["PSModulePath"] = Path.Combine(Environment.SystemDirectory, @"WindowsPowerShell\v1.0\Modules");
        start.EnvironmentVariables["JEV_SETUP_GATE"] = Path.Combine(folder, "start.gate");
        using (ProcessGroup group = new ProcessGroup())
        using (Process process = new Process())
        using (ManualResetEvent outputEnd = new ManualResetEvent(false))
        using (ManualResetEvent errorEnd = new ManualResetEvent(false))
        {
            process.StartInfo = start;
            process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                if (e.Data == null) outputEnd.Set(); else Say(e.Data);
            };
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                if (e.Data == null) errorEnd.Set(); else Say(e.Data);
            };
            ConsoleCancelEventHandler cancel = delegate(object sender, ConsoleCancelEventArgs e) {
                e.Cancel = true;
                Cancelled = true;
                group.Cancel();
            };
            Console.CancelKeyPress += cancel;
            try
            {
                // Do not leak the launcher's original pipe/console handles into
                // descendants. Only the explicitly redirected child pipes inherit.
                foreach (int kind in new int[] { -10, -11, -12 })
                    SetHandleInformation(GetStdHandle(kind), 1, 0);
                process.Start();
                process.StandardInput.Close();
                try { group.Attach(process); }
                catch { if (!process.HasExited) process.Kill(); throw; }
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                // The script waits here until process-tree ownership is established.
                File.WriteAllText(start.EnvironmentVariables["JEV_SETUP_GATE"], "ready");
                while (!process.WaitForExit(250)) { }
                int result = Cancelled ? 1602 : process.ExitCode;
                if (result == 0) group.Preserve(); else group.Cancel();
                // Do not wait forever if a background descendant inherited a pipe.
                outputEnd.WaitOne(1500);
                errorEnd.WaitOne(1500);
                try { process.CancelOutputRead(); } catch (InvalidOperationException) { }
                try { process.CancelErrorRead(); } catch (InvalidOperationException) { }
                return result;
            }
            finally { Console.CancelKeyPress -= cancel; }
        }
    }

    public static int Main(string[] args)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        bool unattended = Array.Exists(args, delegate(string a) { return a == "--non-interactive"; });
        string folder = null;
        int result = 1;
        try
        {
            List<string> options = new List<string>();
            bool verify = false;
            HashSet<string> switches = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {
                "-DependenciesOnly", "-NoLaunch", "-NoBrowser", "-NoShortcuts"
            };
            HashSet<string> values = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {
                "-InstallDir", "-DownloadCache", "-ModelCache"
            };
            for (int i = 0; i < args.Length; i++)
            {
                string option = args[i];
                if (option == "--non-interactive") continue;
                if (option == "--verify") { verify = true; continue; }
                if (option == "--help" || option == "/?")
                {
                    Say("Jev QAT E2B Setup " + Version + " - Windows x64");
                    Say("Doble clic: instala dependencias, E2B QAT Q4 y su proyector; abre la interfaz.");
                    Say("Opciones: -InstallDir ruta, -DownloadCache ruta, -ModelCache ruta,");
                    Say("-DependenciesOnly, -NoLaunch, -NoBrowser, -NoShortcuts, --verify, --non-interactive");
                    return 0;
                }
                if (switches.Contains(option)) { options.Add(option); continue; }
                if (values.Contains(option) && i + 1 < args.Length && !args[i + 1].StartsWith("-"))
                {
                    options.Add(option); options.Add(args[++i]); continue;
                }
                throw new ArgumentException("Opcion no valida o sin valor: " + option);
            }
            if (verify && options.Count != 0) throw new ArgumentException("--verify no acepta opciones de instalacion.");
            byte[] payload = Payload();
            using (ZipArchive zip = new ZipArchive(new MemoryStream(payload), ZipArchiveMode.Read)) Validate(zip);
            if (verify)
            {
                Say("PAYLOAD_OK version=" + Version + " sha256=" + PayloadHash);
                return 0;
            }
            string logs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "JevQatLocal-Setup", "logs");
            Directory.CreateDirectory(logs);
            string logPath = Path.Combine(logs, DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N") + ".log");
            Log = new StreamWriter(logPath, false, new UTF8Encoding(false));
            Say("Instalador de E2B QAT para Windows - v" + Version);
            Say("Descarga inicial aproximada: 4,5 GB. Espacio de disco recomendado: 8 GB.");
            Say("Registro: " + logPath);
            Say("Puedes cancelar con Ctrl+C y volver a ejecutar para reanudar el modelo.");
            folder = Extract(payload);
            result = Run(folder, options);
            Say(result == 0 ? "Instalacion completada." : Cancelled ? "Instalacion cancelada. Puedes reanudarla." : "La instalacion no se completo. Consulta el registro y vuelve a intentarlo.");
        }
        catch (Exception error) { Say("ERROR: " + error.Message); }
        finally
        {
            Cleanup(folder);
            lock (OutputLock) { if (Log != null) { Log.Dispose(); Log = null; } }
        }
        if (result != 0 && !Cancelled && !unattended && !Console.IsInputRedirected)
        {
            Console.WriteLine("Pulsa una tecla para cerrar.");
            Console.ReadKey(true);
        }
        return result;
    }
}

// Closing an incomplete installer kills only its own process tree. On success,
// remove KILL_ON_JOB_CLOSE so the installed API/backend outlive the setup EXE.
internal sealed class ProcessGroup : IDisposable
{
    private IntPtr handle;
    private bool completed;
    private readonly object sync = new object();
    [StructLayout(LayoutKind.Sequential)] private struct BasicLimits {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] private struct Counters {
        public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] private struct Limits {
        public BasicLimits Basic;
        public Counters IO;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr security, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int kind, ref Limits limits, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint code);
    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr value);

    public ProcessGroup()
    {
        handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        try { Configure(0x2000); } catch { CloseHandle(handle); handle = IntPtr.Zero; throw; }
    }
    private void Configure(uint flags)
    {
        Limits limits = new Limits();
        limits.Basic.Flags = flags;
        if (!SetInformationJobObject(handle, 9, ref limits, (uint)Marshal.SizeOf(typeof(Limits))))
            throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    public void Attach(Process process)
    {
        if (!AssignProcessToJobObject(handle, process.Handle)) throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    public void Preserve() { lock (sync) { Configure(0); completed = true; } }
    public void Cancel() { lock (sync) { if (handle != IntPtr.Zero && !completed) TerminateJobObject(handle, 1602); } }
    public void Dispose()
    {
        lock (sync) { if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; } }
    }
}
