unit testreceiveservices;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  servicemanagers;

type

  TTestReceiveService= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestHookUp; 
  end; 

implementation

procedure TTestReceiveService.TestHookUp; 
begin
  Fail('Write your own test'); 
end; 

procedure TTestReceiveService.SetUp; 
begin

end; 

procedure TTestReceiveService.TearDown; 
begin

end; 

initialization

  RegisterTest(TTestReceiveService); 
end.

