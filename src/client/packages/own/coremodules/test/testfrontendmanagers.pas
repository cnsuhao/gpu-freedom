unit testfrontendmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  frontendmanagers;

type

  TTestFrontendManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestFrontendManager;
  private
    frontman_ : TFrontendManager;
    sQ_       : TRegisterQueue;
    bQ_       : TRegisterQueue;
  end;

implementation

procedure TTestFrontendManager.TestFrontendManager;
var info  : TRegisterInfo;
    start : Longint;
begin
  info := frontman_.prepareRegisterInfo4Core('a');
  bQ_.registerJob(info);
  info := frontman_.prepareRegisterInfo4FileFrontend('a','/workunits/',
          'result.txt', 'simclimate.exe','TSimClimateForm',
          'Simulation climate frontend');
  bQ_.registerJob(info);
  info := frontman_.prepareRegisterInfo4UdpFrontend('a','127.0.0.1',23456,
            'orsa.exe', 'TOrbitSim', 'Orbit Simulation and Reconstruction frontend');
  bQ_.registerJob(info);

  start := 1;
  AssertEquals('Found register info for core', true, bQ_.findMultipleRI4Job('a',info, start));
  if info.typeID <> ct_None then raise Exception.Create('Core is of wrong type');
  AssertEquals('Found register info for file frontend', true, bQ_.findMultipleRI4Job('a',info, start));
  if info.typeID <> ct_Files then raise Exception.Create('File Frontend is of wrong type');
  AssertEquals('File frontend exe name', 'simclimate.exe', info.executable);
  AssertEquals('Found register info for udp frontend', true, bQ_.findMultipleRI4Job('a',info, start));
  if info.typeID <> ct_Udp_IP then raise Exception.Create('UDP Frontend is of wrong type');
  AssertEquals('UDP frontend exe name', 'orsa.exe', info.executable);
  AssertEquals('No more info found', false, bQ_.findMultipleRI4Job('a',info, start));
end;

procedure TTestFrontendManager.SetUp; 
begin
  frontman_ := TFrontendManager.Create();
  sQ_ := frontman_.getStandardQueue();
  bQ_ := frontman_.getBroadcastQueue();
end; 

procedure TTestFrontendManager.TearDown; 
begin
  frontman_.Free;
end; 

initialization

  RegisterTest(TTestFrontendManager); 
end.

