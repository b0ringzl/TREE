using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Windows;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;

namespace SpaceScan.Trees.Desktop;

public partial class MainWindow : Window
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(8) };
    private static readonly object LogGate = new();
    private readonly string _logPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "SpaceScan", "Trees", "logs", "desktop-launcher.log");
    private bool _starting;
    private Process? _portableSupervisor;
    private string? _portableRoot;
    private bool _portableReady;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += async (_, _) => await StartAsync();
        Closing += (_, _) => StopOwnedPortableService();
    }

    private async void Retry_Click(object sender, RoutedEventArgs e) => await StartAsync();

    private async Task StartAsync()
    {
        if (_starting) return;
        _starting = true;
        Retry.Visibility = Visibility.Collapsed;
        Progress.Visibility = Visibility.Visible;
        SetStartupStage("Starting recognition services…");
        Log("Desktop startup requested.");
        try
        {
            var portable = FindPortableRoot();
            string url;
            if (portable != null)
                url = await RunPortableAsync(portable);
            else
            {
                var root = FindProjectRoot();
                var script = Path.Combine(root, "spacescan_trees_demo_v1", "start-live.ps1");
                await RunLauncherAsync(script, root);
                url = "http://localhost:8043/image-only";
            }
            SetStartupStage("Opening recognition workspace…");
            await Browser.EnsureCoreWebView2Async().WaitAsync(TimeSpan.FromSeconds(35));
            Browser.CoreWebView2.NewWindowRequested -= OpenPopup;
            Browser.CoreWebView2.NewWindowRequested += OpenPopup;
            var loaded = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            void OnNavigated(object? sender, CoreWebView2NavigationCompletedEventArgs e)
            {
                Browser.NavigationCompleted -= OnNavigated;
                loaded.TrySetResult(e.IsSuccess);
            }
            Browser.NavigationCompleted += OnNavigated;
            Browser.Source = new Uri(url);
            bool navigated;
            try { navigated = await loaded.Task.WaitAsync(TimeSpan.FromSeconds(25)); }
            finally { Browser.NavigationCompleted -= OnNavigated; }
            if (!navigated)
                throw new IOException("The local recognition page did not load. Check the startup log.");
            Log("Desktop window loaded the local recognition page.");
            Browser.Visibility = Visibility.Visible;
            StartupPanel.Visibility = Visibility.Collapsed;
        }
        catch (Exception error)
        {
            StopOwnedPortableService();
            Log(error.ToString());
            Status.Text = $"The desktop app could not start.\n{error.Message}\n\nLog: {_logPath}";
            Progress.Visibility = Visibility.Collapsed;
            Retry.Visibility = Visibility.Visible;
        }
        finally { _starting = false; }
    }

    private void SetStartupStage(string message)
    {
        Status.Text = message;
    }

    private static string? FindPortableRoot()
    {
        var folder = AppContext.BaseDirectory;
        return File.Exists(Path.Combine(folder, "launch.py")) &&
               File.Exists(Path.Combine(folder, "runtime", "python.exe")) &&
               File.Exists(Path.Combine(folder, "app", "spacescan_trees_demo_v1", "server.py"))
            ? folder : null;
    }

    private async Task<string> RunPortableAsync(string root)
    {
        var start = new ProcessStartInfo(Path.Combine(root, "runtime", "python.exe"))
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var argument in new[] { "-s", Path.Combine(root, "launch.py"), "--edition", "image-only", "--no-browser" })
            start.ArgumentList.Add(argument);
        start.Environment["SAM_PYTHON"] = Path.Combine(root, "runtime", "python.exe");
        start.Environment["SPACESCAN_PRIVATE_PACKAGE"] = "1";
        var settingsPath = Path.Combine(root, "desktop-settings.json");
        if (File.Exists(settingsPath))
        {
            using var settings = JsonDocument.Parse(File.ReadAllText(settingsPath));
            if (settings.RootElement.TryGetProperty("vmms_root", out var vmms) &&
                vmms.ValueKind == JsonValueKind.String && Directory.Exists(vmms.GetString()))
                start.Environment["SPACESCAN_VMMS_ROOT"] = vmms.GetString();
            if (settings.RootElement.TryGetProperty("dropbox_app_key", out var dropbox) &&
                dropbox.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(dropbox.GetString()))
                start.Environment["SPACESCAN_DROPBOX_APP_KEY"] = dropbox.GetString();
            if (settings.RootElement.TryGetProperty("onedrive_client_id", out var onedrive) &&
                onedrive.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(onedrive.GetString()))
                start.Environment["SPACESCAN_ONEDRIVE_CLIENT_ID"] = onedrive.GetString();
        }
        var process = Process.Start(start) ?? throw new IOException("Unable to start the bundled services.");
        _portableRoot = root;
        _portableSupervisor = process;
        _portableReady = false;
        var ready = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        _ = Task.Run(async () =>
        {
            while (await process.StandardOutput.ReadLineAsync() is { } line)
            {
                Log("portable: " + line);
                if (line.StartsWith("Preparing bundled runtime", StringComparison.Ordinal))
                    _ = Dispatcher.InvokeAsync(() => SetStartupStage("Preparing the bundled runtime…"));
                else if (line.StartsWith("Waiting for the previous desktop session", StringComparison.Ordinal))
                    _ = Dispatcher.InvokeAsync(() => SetStartupStage("Waiting for the previous session to close…"));
                else if (line.StartsWith("Checking model files", StringComparison.Ordinal))
                    _ = Dispatcher.InvokeAsync(() => SetStartupStage("Checking installed model files…"));
                else if (line.StartsWith("GPU:", StringComparison.Ordinal))
                    _ = Dispatcher.InvokeAsync(() => SetStartupStage("Loading the recognition model…"));
                if (line.StartsWith("READY ", StringComparison.Ordinal))
                {
                    _portableReady = true;
                    ready.TrySetResult(line[6..].Replace("127.0.0.1", "localhost"));
                }
                else if (line.StartsWith("Already running: ", StringComparison.Ordinal))
                    ready.TrySetResult(line[17..].Replace("127.0.0.1", "localhost"));
                else if (line.StartsWith("ERROR:", StringComparison.Ordinal))
                    ready.TrySetException(new InvalidOperationException(line));
            }
            if (!ready.Task.IsCompleted)
                ready.TrySetException(new IOException($"Bundled services exited with code {process.ExitCode}. See {_logPath}."));
        });
        _ = Task.Run(async () =>
        {
            while (await process.StandardError.ReadLineAsync() is { } line)
                Log("portable stderr: " + line);
        });
        try
        {
            var url = await ready.Task.WaitAsync(TimeSpan.FromMinutes(8));
            var health = new Uri(new Uri(url), "/api/health");
            using var response = await Http.GetAsync(health);
            response.EnsureSuccessStatusCode();
            return url;
        }
        catch
        {
            if (!process.HasExited)
                StopOwnedPortableService();
            throw;
        }
    }

    private void StopOwnedPortableService()
    {
        var process = _portableSupervisor;
        var root = _portableRoot;
        if (process == null || root == null || process.HasExited) return;
        try
        {
            if (_portableReady)
            {
                File.WriteAllText(Path.Combine(root, "user_data", "stop.request"), "");
                Log("Requested shutdown of this desktop window's bundled service.");
            }
            else
            {
                process.Kill(entireProcessTree: true);
                Log("Stopped this window's unfinished startup process.");
            }
        }
        catch (Exception error) { Log("Shutdown request failed: " + error); }
    }

    private static string FindProjectRoot()
    {
        var explicitRoot = Environment.GetEnvironmentVariable("SPACESCAN_PROJECT_ROOT");
        if (!string.IsNullOrWhiteSpace(explicitRoot) &&
            File.Exists(Path.Combine(explicitRoot, "spacescan_trees_demo_v1", "start-live.ps1")))
            return Path.GetFullPath(explicitRoot);
        foreach (var origin in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            for (var folder = new DirectoryInfo(origin); folder != null; folder = folder.Parent)
                if (File.Exists(Path.Combine(folder.FullName, "spacescan_trees_demo_v1", "start-live.ps1")))
                    return folder.FullName;
        }
        throw new FileNotFoundException("Demo project was not found. Set SPACESCAN_PROJECT_ROOT to the project folder.");
    }

    private async Task RunLauncherAsync(string script, string root)
    {
        var start = new ProcessStartInfo("powershell.exe")
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var argument in new[] { "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Edition", "image-only" })
            start.ArgumentList.Add(argument);
        using var process = Process.Start(start) ?? throw new IOException("Unable to start the local service launcher.");
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(4));
        var stdout = process.StandardOutput.ReadToEndAsync(timeout.Token);
        var stderr = process.StandardError.ReadToEndAsync(timeout.Token);
        try { await process.WaitForExitAsync(timeout.Token); }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("The local services did not become ready within four minutes.");
        }
        var output = await stdout;
        var errors = await stderr;
        Log($"Launcher exit={process.ExitCode}\n{output}\n{errors}");
        if (process.ExitCode != 0)
            throw new InvalidOperationException(string.IsNullOrWhiteSpace(errors) ? output.Trim() : errors.Trim());
        using var response = await Http.GetAsync("http://127.0.0.1:8043/api/health");
        response.EnsureSuccessStatusCode();
    }

    private async void OpenPopup(object? sender, CoreWebView2NewWindowRequestedEventArgs e)
    {
        var deferral = e.GetDeferral();
        try
        {
            var popupBrowser = new WebView2();
            var popup = new Window
            {
                Title = "SpaceScan Trees · Sign in / Choose files",
                Width = 900, Height = 720, MinWidth = 600, MinHeight = 450,
                Content = popupBrowser, Owner = this, WindowStartupLocation = WindowStartupLocation.CenterOwner,
            };
            await popupBrowser.EnsureCoreWebView2Async(Browser.CoreWebView2.Environment);
            e.NewWindow = popupBrowser.CoreWebView2;
            popup.Show();
        }
        catch (Exception error) { Log($"Popup failed: {error}"); e.Handled = true; }
        finally { deferral.Complete(); }
    }

    private void Log(string message)
    {
        lock (LogGate)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_logPath)!);
            File.AppendAllText(_logPath, $"[{DateTimeOffset.Now:O}] {message}\n");
        }
    }
}
