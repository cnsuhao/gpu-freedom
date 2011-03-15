program gpu;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, mainapp, SQLDBLaz, netmapperforms,
chatforms;

begin
  Application.Initialize;
  Application.CreateForm(TGPUMainApp, GPUMainApp);
  Application.CreateForm(TNetmapperForm, NetmapperForm);
  Application.CreateForm(TChatForm, ChatForm);
  Application.Run;
end.

