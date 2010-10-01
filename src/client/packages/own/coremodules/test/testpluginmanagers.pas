unit testpluginmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  pluginmanagers, loggers, stacks;

type

  TTestPluginManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestPluginManager;
  private
    logger_  : TLogger;
    plugman_ : TPluginManager;
  end; 

implementation

procedure TTestPluginManager.TestPluginManager;
var error : TStkError;
begin
  clearError(error);
  AssertEquals('Basic plugin is not loaded', false, plugman_.isLoaded('basic'));
  AssertEquals('Loading basic plugin', true, plugman_.loadOne('basic', error));
  AssertEquals('No error while loading plugin', true, error.errorId=0);
  AssertEquals('Basic plugin is loaded', true, plugman_.isLoaded('basic'));
  //AssertEquals('asdfasf plugin is not loaded', false, plugman_.isLoaded('asdfasf'));

  {
  plugman_.discardAll();
  plugman_.loadAll(error);
  AssertEquals('No error while loading plugins', 0, error.errorID);
  AssertEquals('basic plugin is loaded', true, plugman_.isLoaded('basic'));
  AssertEquals('strbasic plugin is loaded', true, plugman_.isLoaded('strbasic'));
  AssertEquals('asdfasdffsd plugin is not loaded', false, plugman_.isLoaded('asdfasdffsd'));
  }
end;

procedure TTestPluginManager.SetUp; 
var path,
    extension : String;
begin
 path            := ExtractFilePath(ParamStr(0));
 extension       := 'dll';
 logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log');
 plugman_        := TPluginManager.Create(path+PathDelim+'plugins'+PathDelim+'lib', extension, logger_);
end;

procedure TTestPluginManager.TearDown; 
begin
 plugman_.Free;
 logger_.Free;
end; 

initialization

  RegisterTest(TTestPluginManager); 
end.

