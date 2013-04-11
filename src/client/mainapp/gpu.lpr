program gpu;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, mainapp, SQLDBLaz, netmapperforms,
  chatforms, parametersforms, coreobjects, jobmanagementforms;

begin
  Application.Initialize;
  loadCoreObjects('gpugui', 'GPU GUI', -1);
  Application.CreateForm(TGPUMainApp, GPUMainApp);
  Application.CreateForm(TNetmapperForm, NetmapperForm);
  Application.CreateForm(TChatForm, ChatForm);
  Application.CreateForm(TParametersForm, ParametersForm);
  Application.CreateForm(TJobManagementForm, JobManagementForm);
  Application.Run;
  discardCoreObjects;
end.

