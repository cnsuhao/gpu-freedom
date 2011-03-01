program gpu;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, mainapp;

begin
  Application.Initialize;
  Application.CreateForm(TGPUMainApp, GPUMainApp);
  Application.Run;
end.

