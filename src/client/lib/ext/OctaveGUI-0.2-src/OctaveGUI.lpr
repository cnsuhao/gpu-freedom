program OctaveGUI;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, FormMain, LResources;

{$IFDEF WINDOWS}{$R OctaveGUI.rc}{$ENDIF}

begin
  {$I OctaveGUI.lrs}
  Application.Title:='Octave GUI';
  Application.Initialize;
  Application.CreateForm(TMainForm,MainForm);
  Application.Run;
end.

