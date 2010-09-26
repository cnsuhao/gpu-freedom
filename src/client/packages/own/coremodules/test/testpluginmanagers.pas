unit testpluginmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  pluginmanagers;

type

  TTestPluginManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestHookUp; 
  end; 

implementation

procedure TTestPluginManager.TestHookUp; 
begin
end;

procedure TTestPluginManager.SetUp; 
begin

end; 

procedure TTestPluginManager.TearDown; 
begin

end; 

initialization

  RegisterTest(TTestPluginManager); 
end.

