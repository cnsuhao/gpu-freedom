program gpu;

{$mode objfpc}{$H+}
{ $DEFINE COREINCLUDED}

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
  coremonitor := TCoreMonitor.Create(0);
  {$IFNDEF COREINCLUDED}
  coremonitor.startCore;
  {$ENDIF}
  loadCoreObjects('gpugui', 'GPU GUI', -1);
  Application.CreateForm(TGPUMainApp, GPUMainApp);
  Application.CreateForm(TNetmapperForm, NetmapperForm);
  Application.CreateForm(TChatForm, ChatForm);
  Application.CreateForm(TParametersForm, ParametersForm);
  Application.Run;

  discardCoreObjects;
  {$IFNDEF COREINCLUDED}
  coremonitor.stopCore;
  {$ENDIF}
  coremonitor.Free;
end.

