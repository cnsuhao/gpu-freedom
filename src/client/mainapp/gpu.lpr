program gpu;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, mainapp, SQLDBLaz, netmapperforms,
  chatforms, parametersforms, coreobjects, coremonitors;

var coremonitor : TCoreMonitor;

begin
  Application.Initialize;
  coremonitor := TCoreMonitor.Create();
  coremonitor.startCore;
  loadCoreObjects('gpugui');
  Application.CreateForm(TGPUMainApp, GPUMainApp);
  Application.CreateForm(TNetmapperForm, NetmapperForm);
  Application.CreateForm(TChatForm, ChatForm);
  Application.CreateForm(TParametersForm, ParametersForm);
  Application.Run;

  discardCoreObjects;
  coremonitor.stopCore;
  coremonitor.Free;
end.

