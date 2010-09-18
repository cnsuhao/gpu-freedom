unit simclimateUnit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs, climaconstants,
  ExtCtrls, StdCtrls, mainloop, datastructure,  earthDrawingControl, conversion, averages, initmodel, plotBlueBall,
  OpenGLContext, GL, GLU, threeDplots, OpenGLWorldControl, ParametersForm;

type
  { TearthForm }
  TearthForm = class(TForm)
    paramButton: TButton;
    cbPlotEachHour: TCheckBox;
    cbRotate: TCheckBox;
    cbSphere: TCheckBox;
    DayLabel: TLabel;
    gbSelection: TGroupBox;
    gbLine: TGroupBox;
    gbTime: TGroupBox;
    gbStatus: TGroupBox;
    HourLabel: TLabel;
    lblAtmTemp: TLabel;
    lblDay: TLabel;
    lblHour: TLabel;
    lblHumidity: TLabel;
    lblIceSquaresNorth: TLabel;
    lblIceSquaresSouth: TLabel;
    lblIceSquaresTotal: TLabel;
    lblOceanTemp: TLabel;
    lblOverallTemp: TLabel;
    lblSurfaceTemp: TLabel;
    lblYear: TLabel;
    rbCO2: TRadioButton;
    rbAshes: TRadioButton;
    rbClouds: TRadioButton;
    rbLine1: TRadioButton;
    rbLine2: TRadioButton;
    rbMarineCurrents: TRadioButton;
    rbOverview: TRadioButton;
    rbSurface: TRadioButton;
    rbTempAtmosphere: TRadioButton;
    rbTempSurface: TRadioButton;
    rbWater: TRadioButton;
    rbWaterVegetation: TRadioButton;
    rbWinds: TRadioButton;
    startButton: TButton;
    eDrawingControl,
    eDrawingControl2    : TEarthDrawingControl;
    stopButton: TButton;
    OpenGLWorldControl : TOpenGLWorldControl;
    TAtmLabel: TLabel;
    THumidityLabel: TLabel;
    TIceSquaresLabel: TLabel;
    TIceSquaresNorthLabel: TLabel;
    TIceSquaresSouthLabel: TLabel;
    TOceanLabel: TLabel;
    TOverallLabel: TLabel;
    TSurfaceLabel: TLabel;
    YearLabel: TLabel;

    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure rbLine1Change(Sender: TObject);
    procedure rbLine2Change(Sender: TObject);

    procedure startButtonClick(Sender: TObject);
    procedure stopButtonClick(Sender: TObject);
    procedure paramButtonClick(Sender: TObject);

    procedure updateControls(Sender : TObject);
    procedure onAppIdle(Sender: TObject; var Done: Boolean);

    procedure cbRotateClick(Sender: TObject);
    procedure cbSphereClick(Sender: TObject);

  private
    { private declarations }
    stopSimulation  : Boolean;
    pDrawingControl : PEarthDrawingControl;

    procedure paintControls;
    procedure initEarthModel;
    procedure setPlotCheckbox(pControl : PEarthDrawingControl);

  end; 

var
  earthForm: TearthForm;

implementation

var
    clima : TClima;
    world : TWorld;
    t     : TTime;
    s     : TSolarSurface;
    tmpGrid : TGrid;

{ TearthForm }

procedure TEarthForm.updateControls(Sender : TObject);
begin
  if rbOverview.Checked then
     pDrawingControl^.setPlotMode(PAINT_EARTH_ATMOSPHERE)
  else
  if rbTempAtmosphere.Checked then
     pDrawingControl^.setPlotMode(PAINT_TEMPERATURE_ATM)
  else
  if rbTempSurface.Checked then
     pDrawingControl^.setPlotMode(PAINT_TEMPERATURE_SURFACE)
  else
  if rbWinds.Checked then
     pDrawingControl^.setPlotMode(PAINT_WIND)
  else
  if rbMarineCurrents.Checked then
     pDrawingControl^.setPlotMode(PAINT_MARINE_CURRENTS)
  else
  if rbSurface.Checked then
     pDrawingControl^.setPlotMode(PAINT_SURFACE_TRANSFER)
  else
  if rbWaterVegetation.Checked then
     pDrawingControl^.setPlotMode(PAINT_WATER_AND_VEGETATION)
  else
  if rbWater.Checked then
     pDrawingControl^.setPlotMode(PAINT_WATER)
  else
  if rbClouds.Checked then
     pDrawingControl^.setPlotMode(PAINT_CLOUDS)
  else
  if rbAshes.Checked then
     pDrawingControl^.setPlotMode(PAINT_ASHES)
  else
  if rbCO2.Checked then
     pDrawingControl^.setPlotMode(PAINT_CO2);

  paintControls;
end;

procedure TEarthForm.setPlotCheckbox(pControl : PEarthDrawingControl);
begin
  if pControl^.getPlotMode = PAINT_EARTH_ATMOSPHERE then
     rbOverview.Checked := true
  else
  if pControl^.getPlotMode = PAINT_TEMPERATURE_ATM then
     rbTempAtmosphere.Checked := true
  else
  if pControl^.getPlotMode = PAINT_TEMPERATURE_SURFACE then
     rbTempSurface.Checked := true
  else
  if pControl^.getPlotMode = PAINT_WIND then
     rbWinds.Checked := true
  else
  if pControl^.getPlotMode = PAINT_MARINE_CURRENTS then
     rbMarineCurrents.Checked := true
  else
  if pControl^.getPlotMode = PAINT_SURFACE_TRANSFER then
     rbMarineCurrents.Checked := true
  else
  if pControl^.getPlotMode = PAINT_WATER_AND_VEGETATION then
     rbMarineCurrents.Checked := true
  else
  if pControl^.getPlotMode = PAINT_WATER then
     rbWater.Checked := true
  else
  if pControl^.getPlotMode = PAINT_CLOUDS then
     rbClouds.Checked := true
  else
  if pControl^.getPlotMode = PAINT_ASHES then
     rbAshes.Checked  := true
  else
  if pControl^.getPlotMode = PAINT_CO2 then
     rbCO2.Checked  := true;

end;

procedure TearthForm.paintControls;
var isDone : Boolean;
    colors : PGridColor;

begin
  lblHour.Caption  := FloatToStr(t.hour);

  eDrawingControl.paintEarth(world, clima, s);
  eDrawingControl2.paintEarth(world, clima, s);

  eDrawingControl.Paint;
  eDrawingControl2.Paint;

  lblOverallTemp.Caption := FloatToStr(KtoC(computeAvgKTemperature(world, clima, AVERAGE)));
  lblAtmTemp.Caption := FloatToStr(KtoC(computeAvgKTemperature(world, clima, ATMOSPHERE)));
  lblOceanTemp.Caption := FloatToStr(KtoC(computeAvgKTemperature(world, clima, OCEAN)));
  lblSurfaceTemp.Caption := FloatToStr(KtoC(computeAvgKTemperature(world, clima, TERRAIN)));
  lblHumidity.Caption := FloatToStr(computeAvgHumidity(world, clima));
  lblIceSquaresTotal.Caption := IntToStr(computeIceCoverage(clima, NONE));
  lblIceSquaresNorth.Caption := IntToStr(computeIceCoverage(clima, NORTH));
  lblIceSquaresSouth.Caption := IntToStr(computeIceCoverage(clima, SOUTH));

  if rbLine1.Checked then
      colors := eDrawingControl.getColors
    else
  if rbLine2.Checked then
      colors := eDrawingControl2.getColors;

  OpenGLWorldControl.setColors(colors);
  onAppIdle(self, isDone);

end;

procedure TearthForm.startButtonClick(Sender: TObject);
var
    day, hour, year : Longint;

begin
 stopButton.Enabled  := True;
 startButton.Enabled := False;

 stopSimulation := False;
 for year:=2000 to 3000 do
  for day := 1 to 365  do
  begin
    lblYear.Caption := IntToStr(t.year);
    lblDay.Caption  := IntToStr(t.day);
    for hour := 1 to 24*Trunc(15/TSimConst.degree_step) do
       begin
         if stopSimulation then Exit;
         if ParamForm.isInitialConditionsChanged then
                                   begin
                                     initEarthModel;
                                     paramForm.resetInitialConditionsChanged;
                                   end;

         mainSimTimeStepLoop(world, clima, s, t, tmpGrid);
         if cbPlotEachHour.Checked then paintControls;
         Application.ProcessMessages;
       end;

       mainSimDayLoop(world, clima, s, t, tmpGrid);
       if not cbPlotEachHour.Checked then paintControls;
   end;

end;

procedure TearthForm.stopButtonClick(Sender: TObject);
begin
 stopSimulation := True;
 stopButton.Enabled  := False;
 startButton.Enabled := True;
end;



procedure TearthForm.FormCreate(Sender: TObject);
begin
  initWorld(world, '');
  initEarthModel;

  eDrawingControl:= TEarthDrawingControl.Create(Self, @world, @clima, @s, @t, PAINT_EARTH_ATMOSPHERE, 1);
  eDrawingControl.Top := 0;
  eDrawingControl.Left := 0;
  eDrawingControl.Parent := Self;
  eDrawingControl.DoubleBuffered := True;

  eDrawingControl2:= TEarthDrawingControl.Create(Self, @world, @clima, @s, @t, PAINT_TEMPERATURE_ATM, 1);
  eDrawingControl2.Top := 0;
  eDrawingControl2.Left := 360 + 15;
  eDrawingControl2.Parent := Self;
  eDrawingControl2.DoubleBuffered := True;

  OpenGLWorldControl:=TOpenGLWorldControl.Create(Self);
  with OpenGLWorldControl do begin
    Name:='OpenGLWorldControl';
    Align:=alNone;
    Parent:=Self;
    Top := 180+15+30;
    Left := 0;
    Height := earthForm.Height - Top - 15;
    Width := 2 * 360 + 15;
  end;
  OpenGLWorldControl.setColors(eDrawingControl.getColors);
  OpenGLWorldControl.setParameters(world, clima);

  Application.AddOnIdleHandler(@OnAppIdle);
  pDrawingControl := @eDrawingControl;
end;

procedure TearthForm.cbRotateClick(Sender: TObject);
begin
  OpenGlWorldControl.setRotate(cbRotate.Checked);
end;

procedure TearthForm.cbSphereClick(Sender: TObject);
begin
  OpenGlWorldControl.setSphere(cbSphere.Checked);
end;

procedure TearthForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  StopSimulation := true;
end;


procedure TearthForm.FormResize(Sender: TObject);
begin
 OpenGlWorldControl.AutoResizeViewport := true;

 if earthForm.Height>640 then
  OpenGLWorldControl.Height := earthForm.Height - OpenGLWorldControl.Top - 15;
 if earthForm.Width > 2*360  + 15 + 210 then
    OpenGLWorldControl.Width := earthForm.Width - 210;

 gbLine.Left := OpenGLWorldControl.Width+15;
 gbStatus.Left := OpenGLWorldControl.Width+15;
 gbTime.Left := OpenGLWorldControl.Width+15;

 OpenGlWorldControl.AutoResizeViewport := false;
end;

procedure TearthForm.rbLine1Change(Sender: TObject);
begin
  if rbLine1.Checked then
          begin
            gbLine.Caption := 'Line 1';
            pDrawingControl := @eDrawingControl;
            setPlotCheckbox(@eDrawingControl);
            paintControls;
          end;
end;

procedure TearthForm.rbLine2Change(Sender: TObject);
begin
   if rbLine2.Checked then
          begin
            gbLine.Caption := 'Line 2';
            pDrawingControl := @eDrawingControl2;
            setPlotCheckbox(@eDrawingControl2);
            paintControls;
          end;
end;

procedure TearthForm.FormDestroy(Sender: TObject);
begin
 eDrawingControl.Free;
 eDrawingControl2.Free;
 OpenGLWorldControl.Free;
end;


procedure TEarthForm.OnAppIdle(Sender: TObject; var Done: Boolean);
begin
  Done:=false;
  OpenGLWorldControl.Invalidate;
end;

procedure TEarthForm.paramButtonClick(Sender: TObject);
begin
  ParamForm.loadSettings;
  ParamForm.visible := True;
end;

procedure TEarthForm.initEarthModel;
begin
  initClima(world, clima, TInitCond.thermic_poles, TInitCond.thermic_poles, '');
  initTime(t, s);
  initPlot;
end;

initialization
  {$I simclimateUnit.lrs}

end.

