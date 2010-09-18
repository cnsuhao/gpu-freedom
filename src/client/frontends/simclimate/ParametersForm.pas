unit ParametersForm;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, climaconstants, MaskEdit, ExtCtrls, ComCtrls, Spin, datastructure,
  conversion, initmodel;

type

  { TParamForm }

  TParamForm = class(TForm)
    btnOK: TButton;
    btnApply: TButton;
    btnCancel: TButton;
    btnReset: TButton;
    cbRotation: TCheckBox;
    cbRevolution: TCheckBox;
    cbInvertWindFlow: TCheckBox;
    cbVulcan: TCheckBox;
    cbNuclearBomb: TCheckBox;
    cbNuclearWar: TCheckBox;
    cbInverseRot: TCheckBox;
    cbPopulation: TCheckBox;
    cbNonlinearGrid: TCheckBox;
    edtEarthRadius: TLabeledEdit;
    gbModel: TGroupBox;
    gbInitialConditions: TGroupBox;
    gbSimulation: TGroupBox;
    gbPhysical: TGroupBox;
    gbSpecial: TGroupBox;
    edtAlbedo: TLabeledEdit;
    edtDistanceFromSun: TLabeledEdit;
    edtDawn: TLabeledEdit;
    edtTauVisible: TLabeledEdit;
    edtTauInfrared: TLabeledEdit;
    edtDistanceLayers: TLabeledEdit;
    edtThermicPoles: TLabeledEdit;
    edtThermicExcursion: TLabeledEdit;
    edtSurfaceShift: TLabeledEdit;
    edtOceanShift: TLabeledEdit;
    edtTerrainShift: TLabeledEdit;
    edtDesertBeltLat: TLabeledEdit;
    edtDesertBeltExt: TLabeledEdit;
    edtDesertBeltTemp: TLabeledEdit;
    edtCloudReflection: TLabeledEdit;
    edtCloudIsolation: TLabeledEdit;
    edtRiverAndLakes: TLabeledEdit;
    edtEnergyTransferWind: TLabeledEdit;
    edtTerrestrialEnergyTemp: TLabeledEdit;
    edtRainHours: TLabeledEdit;
    edtSteamHours: TLabeledEdit;
    edtRadiationHours: TLabeledEdit;
    edtExchangeAtmTerr: TLabeledEdit;
    edtExchangeFluxAtm: TLabeledEdit;
    edtExchangeFluxTerr: TLabeledEdit;
    edtExchangeFluxOcean: TLabeledEdit;
    edtThermicGradientAvg: TLabeledEdit;
    edtThermicGradientsea: TLabeledEdit;
    edtSolarConstant: TLabeledEdit;
    edtEarthInclination: TLabeledEdit;
    edtStefanBoltzmann: TLabeledEdit;
    edtGravAcc: TLabeledEdit;
    edtCpWater: TLabeledEdit;
    edtCpSteam: TLabeledEdit;
    edtCpAir: TLabeledEdit;
    edtCpEarth: TLabeledEdit;
    edtPaintRiverPct: TLabeledEdit;
    edtDecreaseVegetation: TLabeledEdit;
    edtDensityWater: TLabeledEdit;
    edtDensityEarth: TLabeledEdit;
    edtVulcanLat: TLabeledEdit;
    edtVulcanLon: TLabeledEdit;
    edtVulcanHours: TLabeledEdit;
    edtVulcanAshesPct: TLabeledEdit;
    edtNBLat: TLabeledEdit;
    edtNBLon: TLabeledEdit;
    edtNBEnergy: TLabeledEdit;
    edtNBAshesPct: TLabeledEdit;
    edtNBHours: TLabeledEdit;
    edtAshesFalloutPct: TLabeledEdit;
    gbCO2: TGroupBox;
    edtPopIncreasePct: TLabeledEdit;
    edtCO2ProdHuman: TLabeledEdit;
    edtCO2ProdVulcano: TLabeledEdit;
    edtCO2AbsorpVegetation: TLabeledEdit;
    edtCO2AbsorpOcean: TLabeledEdit;
    edtCO2Isolation: TLabeledEdit;
    edtPaintCloudsPct: TLabeledEdit;
    edtEvaporationStart: TLabeledEdit;
    edtFullEvaporation: TLabeledEdit;
    lblEnergySource: TLabel;
    lblStep: TLabel;
    lblAtmLayers: TLabel;
    rbRenewables: TRadioButton;
    rbOil: TRadioButton;
    rb4min: TRadioButton;
    rb20min: TRadioButton;
    rb1h: TRadioButton;
    seAtmLayers: TSpinEdit;

    procedure btnCancelClick(Sender: TObject);
    procedure btnOKClick(Sender: TObject);
    procedure btnResetClick(Sender: TObject);
    procedure btnApplyClick(Sender: TObject);

    procedure onChangeInitialCondition(Sender: TObject);
    function  isInitialConditionsChanged : Boolean;
    procedure resetInitialConditionsChanged;

    procedure FormCreate(Sender : TObject);
    procedure loadSettings;

  private
    { private declarations }
    initialConditionsChanged,
    applyInitialConditions : Boolean;

    function  validateSettings : Boolean;
    function  saveSettings : Boolean;

    function validatePct(field,value : String) : Boolean;
    function validateDegree(field,value : String) : Boolean;
    function validateTemperature(field,value : String) : Boolean;


  end; 

var
  ParamForm: TParamForm;

implementation

{ TParamForm }

procedure TParamForm.btnCancelClick(Sender: TObject);
begin
  Visible := False;

  initialConditionsChanged := false;
  applyInitialConditions   := false;
end;

procedure TParamForm.btnOKClick(Sender: TObject);
begin
  if saveSettings then
            Visible := False;
end;

procedure TParamForm.btnApplyClick(Sender: TObject);
begin
  saveSettings;
end;


procedure TParamForm.btnResetClick(Sender: TObject);
begin
  initModelParameters;
  initConversion(true);
  loadSettings;
end;



procedure TParamForm.FormCreate(Sender : TObject);
begin
  loadSettings;
end;

function TParamForm.validatePct(field,value : String) : Boolean;
var
  _value : Extended;
begin
 _value := StrToFloat(value);
 Result := (_value>=0) and (_value<=1);
 if not Result then ShowMessage('Error in field "'+field+'": value has to be between 0 and 1');
end;

function TParamForm.validateDegree(field,value : String) : Boolean;
var
  _value : Extended;
begin
 _value := StrToFloat(value);
 Result := (_value>=0) and (_value<=360);
 if not Result then ShowMessage('Error in field "'+field+'": value has to be between 0 and 360');
end;

function TParamForm.validateTemperature(field,value : String) : Boolean;
var
  _value : Extended;
begin
 _value := StrToFloat(value);
 Result := (_value>= -TPhysConst.AbsoluteZero);
 if not Result then ShowMessage('Error in field "'+field+'": temperature has to be higher or equal absolute zero');
end;


procedure TParamForm.loadSettings;
begin
 initialConditionsChanged := false;
 applyInitialConditions   := false;

 // model parameters
  edtAlbedo.Text := FloatToStr(TMdlConst.Albedo);
  edtDistanceFromSun.Text := FloatToStr(TMdlConst.distanceFromSun);
  edtTauVisible.Text := FloatToStr(TMdlConst.tau_visible);
  edtTauInfrared.Text := FloatToStr(TMdlConst.tau_infrared);

  cbRotation.Checked := TMdlConst.rotation;
  cbRevolution.Checked := TMdlConst.revolution;
  cbInverseRot.Checked := (TMdlConst.inverse_rotation=-1);

  edtDawn.Text := IntToStr(TMdlConst.initDegreeSunlight);
  edtDistanceLayers.Text := IntToStr(TMdlConst.distance_atm_layers);

  seAtmLayers.Value := TMdlConst.atmospheric_layers;
  seAtmLayers.MaxValue:= MAX_ATM_LAYERS;

  // initial conditions
  edtThermicPoles.Text     := FloatToStr(TInitCond.thermic_poles);
  edtThermicExcursion.Text := FloatToStr(TInitCond.thermic_excursion);
  edtSurfaceShift.Text     := FloatToStr(TInitCond.surface_shift);
  edtOceanShift.Text       := FloatToStr(TInitCond.ocean_shift);
  edtTerrainShift.Text     := FloatToStr(TInitCond.terrain_shift);

  edtDesertBeltLat.Text := FloatToStr(TInitCond.desert_belt_lat);
  edtDesertBeltExt.Text := FloatToStr(TInitCond.desert_belt_ext);
  edtDesertBeltTemp.Text := FloatToStr(TInitCond.desert_belt_delta_T);

  edtThermicGradientAvg.Text := FloatToStr(TInitCond.thermic_gradient_avg);
  edtThermicGradientSea.Text := FloatToStr(TInitCond.thermic_gradient_sea);

  // simulation parameters
  edtCloudReflection.Text := FloatToStr(TSimConst.cloud_reflection_pct);
  edtCloudIsolation.Text  := FloatToStr(TSimConst.cloud_isolation_pct);
  edtCO2Isolation.Text    := FloatToStr(TSimConst.co2_isolation_pct);
  edtRiverAndLakes.Text   := FloatToStr(TSimConst.riverandlakes_pct);
  edtTerrestrialEnergyTemp.Text := FloatToStr(TSimConst.deltaTterrestrialEnergy);

  edtEnergyTransferWind.Text := FloatToStr(TSimConst.pct_wind_transfer);
  cbInvertWindFlow.Checked := (TSimConst.invert_flow = -1);

  edtRainHours.Text         := FloatToStr(TSimConst.rain_hours);
  edtSteamHours.Text        := FloatToStr(TSimConst.steam_hours);
  edtRadiationHours.Text    := FloatToStr(TSimConst.radiation_hours);
  edtExchangeAtmTerr.Text   := FloatToStr(TSimConst.exchange_atm_terr);
  edtExchangeFluxAtm.Text   := FloatToStr(TSimConst.exchange_flux_atm);
  edtExchangeFluxTerr.Text  := FloatToStr(TSimConst.exchange_flux_terrain);
  edtExchangeFluxOcean.Text := FloatToStr(TSimConst.exchange_flux_ocean);
  edtEvaporationStart.Text := FloatToStr(TSimConst.evaporation_start_temp);
  edtFullEvaporation.Text := FloatToStr(TSimConst.full_evaporation_temp);


  if TSimConst.hour_step=1200 then rb20min.Checked := True
  else
  if TSimConst.hour_step=3600 then rb1h.Checked := True
  else
  if TSimConst.hour_step=240 then rb4min.Checked := True;

  edtPaintRiverPct.Text      := FloatToStr(TSimConst.paint_river_pct);
  edtPaintCloudsPct.Text     := FloatToStr(TSimConst.paint_clouds);
  edtDecreaseVegetation.Text := IntToStr(TSimConst.decrease_rain_times);

  // Physical constants
  edtSolarConstant.Text    := FloatToStr(TPhysConst.SolarConstant);
  edtEarthRadius.Text      := FloatToStr(TPhysConst.earth_radius);
  edtEarthInclination.Text := FloatToStr(TPhysConst.earth_inclination_on_ecliptic);
  edtGravAcc.Text          := FloatToStr(TPhysConst.grav_acc);
  edtStefanBoltzmann.Text  := FloatToStr(TPhysConst.stefan_boltzmann);
  edtCpWater.Text          := FloatToStr(TPhysConst.cp_water);
  edtCpSteam.Text          := FloatToStr(TPhysConst.cp_steam);
  edtCpAir.Text            := FloatToStr(TPhysConst.cp_air);
  edtCpEarth.Text          := FloatToStr(TPhysConst.cp_earth);

  edtDensityWater.Text     := FloatToStr(TPhysConst.density_water);
  edtDensityEarth.Text     := FloatToStr(TPhysConst.density_earth);

  // special parameters
  cbVulcan.Checked := TSpecialParam.vulcan;
  edtVulcanLat.Text := IntToStr(TSpecialParam.vulcan_lat);
  edtVulcanLon.Text := IntToStr(TSpecialParam.vulcan_lon);
  edtVulcanHours.Text := FloatToStr(TSpecialParam.vulcan_hours);
  edtVulcanAshesPct.Text := FloatToStr(TSpecialParam.vulcan_ashes_pct);

  cbNuclearBomb.Checked := TSpecialParam.nuclear_bomb;
  cbNuclearWar.Checked  := TSpecialParam.nuclear_war;
  edtNBLat.Text := IntToStr(TSpecialParam.nuclear_bomb_lat);
  edtNBLon.Text := IntToStr(TSpecialParam.nuclear_bomb_lon);
  edtNBEnergy.Text      := FloatToStr(TSpecialParam.nuclear_bomb_energy);
  edtNBAshesPct.Text  := FloatToStr(TSpecialParam.nuclear_ashes_pct);
  edtNBHours.Text := FloatToStr(TSpecialParam.nuclear_war_hours);
  edtAshesFalloutPct.Text := FloatToStr(TSpecialParam.ashes_fallout_pct);

  // CO2 parameters
  cbPopulation.Checked := TSimConst.population;
  if TSimConst.energy_source_oil then
      rbOil.Checked:=true
     else
      rbRenewables.Checked:=true;
  edtPopIncreasePct.Text := FloatToStr(TSimConst.population_increase_pct);
  edtCO2ProdHuman.Text := FloatToStr(TSimConst.co2_production_per_human_per_year);
  edtCO2ProdVulcano.Text := FloatToStr(TSimConst.co2_production_vulcano);
  edtCO2AbsorpVegetation.Text := FloatToStr(TSimConst.co2_absorption_vegetation);
  edtCO2AbsorpOcean.Text := FloatToStr(TSimConst.co2_absorption_ocean);

  cbNonLinearGrid.Checked := isNonLinearConversion;
end;

function TParamForm.validateSettings : Boolean;
begin
  Result := False;
  // model parameters
  if not validatePct('Albedo', edtAlbedo.Text) then Exit;
  if not validatePct('Tau visible', edtTauVisible.Text) then Exit;
  if not validatePct('Tau infrared', edtTauInfrared.Text) then Exit;
  if not validateDegree('Initial Dawn degree', edtDawn.Text) then Exit;

    // initial conditions
  if not validateTemperature('Temperature at poles', edtThermicPoles.Text) then Exit;

  // simulation parameters
  if not validatePct('Cloud reflection', edtCloudReflection.Text) then Exit;
  if not validatePct('Cloud isolation', edtCloudIsolation.Text) then Exit;
  if not validatePct('CO2 isolation', edtCO2isolation.Text) then Exit;
  if not validatePct('River flow percent', edtRiverAndLakes.Text) then Exit;
  if not validatePct('Percent energy transferred by wind', edtEnergyTransferWind.Text) then Exit;

  // physical constants

  // special parameters
  if not validatePct('Vulcan Intensity', edtVulcanAshesPct.Text) then Exit;
  if not validatePct('Nuclear Ashes Intensity', edtNBAshesPct.Text) then Exit;
  if not validatePct('Ashes Fallout', edtAshesFalloutPct.Text) then Exit;

  Result := True;
end;

function TParamForm.saveSettings : Boolean;
begin
  Result := False;
  if not validateSettings then exit;

  applyInitialConditions := false;
  if initialConditionsChanged then
       begin
         if MessageDlg('Would you like to change the initial conditions (it requires a model reset)? ',
          mtWarning, [mbYes, mbNo], 0) = mrYes then
                  begin
                    // initial conditions
                    TInitCond.thermic_poles := StrToFloat(edtThermicPoles.Text);
                    TInitCond.thermic_excursion := StrToFloat(edtThermicExcursion.Text);
                    TInitCond.surface_shift := StrToFloat(edtSurfaceShift.Text);
                    TInitCond.ocean_shift := StrToFloat(edtOceanShift.Text);
                    TInitCond.terrain_shift := StrToFloat(edtTerrainShift.Text);

                    TInitCond.desert_belt_lat := StrToFloat(edtDesertBeltLat.Text);
                    TInitCond.desert_belt_ext := StrToFloat(edtDesertBeltExt.Text);
                    TInitCond.desert_belt_delta_T := StrToFloat(edtDesertBeltTemp.Text);

                    TInitCond.thermic_gradient_avg := StrToFloat(edtThermicGradientAvg.Text);
                    TInitCond.thermic_gradient_sea := StrToFloat(edtThermicGradientSea.Text);
                    TPhysConst.earth_radius  := StrToFloat(edtEarthRadius.Text);

                    applyInitialConditions := true;
                  end
                else
                  begin
                    initialConditionsChanged := false;
                    LoadSettings;
                    Exit;
                  end;
       end;

  // model parameters
  TMdlConst.Albedo := StrToFloat(edtAlbedo.Text);
  TMdlConst.distanceFromSun := StrToFloat(edtDistanceFromSun.Text);
  TMdlConst.tau_visible := StrToFloat(edtTauVisible.Text);
  TMdlConst.tau_infrared := StrToFloat(edtTauInfrared.Text);
  TMdlConst.rotation := cbRotation.Checked;
  TMdlConst.revolution := cbRevolution.Checked;
  if  cbInverseRot.Checked then TMdlConst.inverse_rotation:=-1
     else TMdlConst.inverse_rotation := 1;

  TMdlConst.initDegreeSunlight := StrToInt(edtDawn.Text);
  TMdlConst.distance_atm_layers := StrToInt(edtDistanceLayers.Text);

  TMdlConst.atmospheric_layers := seAtmLayers.Value;


  // simulation parameters
  TSimConst.cloud_reflection_pct := StrToFloat(edtCloudReflection.Text);
  TSimConst.cloud_isolation_pct  := StrToFloat(edtCloudIsolation.Text);
  TSimConst.co2_isolation_pct  := StrToFloat(edtCO2Isolation.Text);
  TSimConst.riverandlakes_pct    := StrToFloat(edtRiverAndLakes.Text);
  TSimConst.deltaTterrestrialEnergy := StrToFloat(edtTerrestrialEnergyTemp.Text);

  TSimConst.pct_wind_transfer := StrToFloat(edtEnergyTransferWind.Text);
  if (cbInvertWindFlow.Checked) then
         TSimConst.invert_flow := -1
       else
         TSimConst.invert_flow := 1;

  TSimConst.rain_hours            := StrToFloat(edtRainHours.Text);
  TSimConst.steam_hours           := StrToFloat(edtSteamHours.Text);
  TSimConst.radiation_hours       := StrToFloat(edtRadiationHours.Text);
  TSimConst.exchange_atm_terr     := StrToFloat(edtExchangeAtmTerr.Text);
  TSimConst.exchange_flux_atm     := StrToFloat(edtExchangeFluxAtm.Text);
  TSimConst.exchange_flux_terrain := StrToFloat(edtExchangeFluxTerr.Text);
  TSimConst.exchange_flux_ocean   := StrToFloat(edtExchangeFluxOcean.Text);

  TSimConst.paint_river_pct     := StrToFloat(edtPaintRiverPct.Text);
  TSimConst.paint_clouds        := StrToFloat(edtPaintCloudsPct.Text);
  TSimConst.decrease_rain_times := StrToInt(edtDecreaseVegetation.Text);

  TSimConst.evaporation_start_temp := StrToFloat(edtEvaporationStart.Text);
  TSimConst.full_evaporation_temp  := StrToFloat(edtFullEvaporation.Text);

  if rb1h.Checked then
      begin
        TSimConst.hour_step := 3600;
        TSimConst.degree_step := 15
      end
  else
  if rb20min.Checked then
      begin
        TSimConst.hour_step := 1200;
        TSimConst.degree_step := 5
      end
  else
  if rb4min.Checked then
      begin
        TSimConst.hour_step := 240;
        TSimConst.degree_step := 1;
      end
  else raise Exception.Create('Problem in setting simulation step');

  // Physical constants
  TPhysConst.SolarConstant := StrToFloat(edtSolarConstant.Text);
  TPhysConst.earth_inclination_on_ecliptic := StrToFloat(edtEarthInclination.Text);
  TPhysConst.grav_acc := StrToFloat(edtGravAcc.Text);
  TPhysConst.stefan_boltzmann := StrToFloat(edtStefanBoltzmann.Text);
  TPhysConst.cp_water := StrToFloat(edtCpWater.Text);
  TPhysConst.cp_steam := StrToFloat(edtCpSteam.Text);
  TPhysConst.cp_air := StrToFloat(edtCpAir.Text);
  TPhysConst.cp_earth := StrToFloat(edtCpEarth.Text);

  TPhysConst.density_water := StrToFloat(edtDensityWater.Text);
  TPhysConst.density_earth := StrToFloat(edtDensityEarth.Text);

  // special parameters
  TSpecialParam.vulcan       := cbVulcan.Checked;
  TSpecialParam.vulcan_lat   := StrToInt(edtVulcanLat.Text);
  TSpecialParam.vulcan_lon   := StrToInt(edtVulcanLon.Text);
  TSpecialParam.vulcan_hours := StrToFloat(edtVulcanHours.Text);
  TSpecialParam.vulcan_ashes_pct := StrToFloat(edtVulcanAshesPct.Text);

  TSpecialParam.nuclear_bomb := cbNuclearBomb.Checked;
  TSpecialParam.nuclear_war  := cbNuclearWar.Checked;
  TSpecialParam.nuclear_bomb_lat := StrToInt(edtNBLat.Text);
  TSpecialParam.nuclear_bomb_lon := StrToInt(edtNBLon.Text);
  TSpecialParam.nuclear_bomb_energy := StrToFloat(edtNBEnergy.Text);
  TSpecialParam.nuclear_ashes_pct := StrToFloat(edtNBAshesPct.Text);
  TSpecialParam.nuclear_war_hours := StrToFloat(edtNBHours.Text);
  TSpecialParam.ashes_fallout_pct := StrToFloat(edtAshesFalloutPct.Text);

  // CO2 parameters
  TSimConst.population := cbPopulation.Checked;
  TSimConst.energy_source_oil := (rbOil.Checked);

  TSimConst.population_increase_pct := StrToFloat(edtPopIncreasePct.Text);
  TSimConst.co2_production_per_human_per_year := StrToFloat(edtCO2ProdHuman.Text);
  TSimConst.co2_production_vulcano := StrToFloat(edtCO2ProdVulcano.Text);
  TSimConst.co2_absorption_vegetation := StrToFloat(edtCO2AbsorpVegetation.Text);
  TSimConst.co2_absorption_ocean := StrToFloat(edtCO2AbsorpOcean.Text);

  InitConversion(cbNonLinearGrid.Checked);

  Result := true;
end;

procedure TParamForm.onChangeInitialCondition(Sender: TObject);
begin
  initialConditionsChanged := True;
end;

function  TParamForm.isInitialConditionsChanged : Boolean;
begin
  Result := applyInitialConditions;
end;

procedure TParamForm.resetInitialConditionsChanged;
begin
  applyInitialConditions := False;
end;

initialization
  {$I ParametersForm.lrs}


end.

