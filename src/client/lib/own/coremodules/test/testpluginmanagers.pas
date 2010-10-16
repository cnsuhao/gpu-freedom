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

    procedure callFunctions;
  end; 

implementation

procedure TTestPluginManager.callFunctions;
var res      : TStkString;
    stk      : TStack;
    float    : TStkFloat;

begin
  clearStk(stk);
  pushFloat(1, stk);
  pushFloat(2, stk);
  AssertEquals('Executing method add', true, plugman_.method_execute('add', stk));
  popFloat(float, stk);
  AssertEquals('1+2 is', 3, float);

  clearStk(stk);
  pushStr('Virgi ', stk);
  pushStr('is beautiful', stk);
  AssertEquals('Executing method concat', true, plugman_.method_execute('concat', stk));
  popStr(res, stk);
  AssertEquals('The truth is that', 'Virgi is beautiful', res);
end;

procedure TTestPluginManager.TestPluginManager;
var error    : TStkError;
    plugname : TStkString;
    i        : Longint;
begin
  clearError(error);
  AssertEquals('Basic plugin is not loaded', false, plugman_.isLoaded('basic'));
  AssertEquals('Loading basic plugin', true, plugman_.loadOne('basic', error));
  AssertEquals('No error while loading plugin', true, error.errorId=0);
  AssertEquals('Basic plugin is loaded', true, plugman_.isLoaded('basic'));
  AssertEquals('asdfasf plugin is not loaded', false, plugman_.isLoaded('asdfasf'));

  plugman_.discardAll();
  plugman_.loadAll(error);
  AssertEquals('No error while loading plugins', 0, error.errorID);
  AssertEquals('basic plugin is loaded', true, plugman_.isLoaded('basic'));
  AssertEquals('strbasic plugin is loaded', true, plugman_.isLoaded('strbasic'));
  AssertEquals('asdfasdffsd plugin is not loaded', false, plugman_.isLoaded('asdfasdffsd'));
  AssertEquals('Method add exists', true, plugman_.method_exists('add', plugname, error));
  AssertEquals('Method add exists in plugin', 'basic', plugname);
  AssertEquals('Method concat exists', true, plugman_.method_exists('concat', plugname, error));
  AssertEquals('Method add exists in plugin', 'strbasic', plugname);
  AssertEquals('Method sfjslfj does not exist', false, plugman_.method_exists('sfjslfj', plugname, error));
  AssertEquals('Plugin name is empty', '', plugname);

  for i:=1 to 100 do callFunctions;
  plugman_.discardAll();
  plugman_.loadAll(error);
  for i:=1 to 100 do callFunctions;
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

