using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Windows;

namespace SpaceScan.Trees.Desktop;

public partial class App : Application
{
    private Mutex? _singleInstance;

    [DllImport("user32.dll")]
    private static extern bool ShowWindowAsync(IntPtr window, int command);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr window);

    private static void ActivateExistingWindow()
    {
        var current = Process.GetCurrentProcess();
        var executable = Path.GetFullPath(Environment.ProcessPath!);
        foreach (var process in Process.GetProcessesByName(current.ProcessName))
        {
            try
            {
                if (process.Id == current.Id || process.MainWindowHandle == IntPtr.Zero ||
                    !string.Equals(Path.GetFullPath(process.MainModule!.FileName), executable,
                                   StringComparison.OrdinalIgnoreCase)) continue;
                ShowWindowAsync(process.MainWindowHandle, 9); // SW_RESTORE
                SetForegroundWindow(process.MainWindowHandle);
                return;
            }
            catch (Exception) { /* Another process may have exited or denied inspection. */ }
        }
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        // A packaged copy and the development preview must not lock each other.
        var installPath = Path.GetFullPath(AppContext.BaseDirectory).ToUpperInvariant();
        var identity = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(installPath)))[..16];
        _singleInstance = new Mutex(true, "Local\\SpaceScanTreesDesktop-" + identity, out var created);
        if (!created)
        {
            ActivateExistingWindow();
            Shutdown();
            return;
        }
        base.OnStartup(e);
        new MainWindow().Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _singleInstance?.Dispose();
        base.OnExit(e);
    }
}
