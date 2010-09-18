program simclimate;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, simclimateUnit,
  riverandlakes, initmodel, threeDplots, OpenGLWorldControl, ParametersForm,
  vulcanandbombs, co2cycle, mainloop;

begin
  Application.Initialize;
  Application.CreateForm(TearthForm, earthForm);
  Application.CreateForm(TParamForm, ParamForm);
  Application.Run;
end.

