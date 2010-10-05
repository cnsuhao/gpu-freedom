unit testjobparsers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  jobparsers, pluginmanagers, methodcontrollers, loggers,
  frontendmanagers, specialcommands, resultcollectors, jobs,
  stacks, stkconstants;

type

  TTestJobParser= class(TTestCase)
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  published
    procedure TestJobParserFloats;
    procedure TestJobParserException;

  private
    plugman_  : TPluginManager;
    meth_     : TMethodController;
    logger_   : TLogger;
    res_      : TResultCollector;
    frontman_ : TFrontendManager;
    spec_     : TSpecialCommand;
    job_      : TJob;

    procedure execJob(job : String);
  end;

implementation

procedure TTestJobParser.execJob(job : String);
var jparser : TJobParser;
begin
 clearStk(job_.stack);
 job_.hasError:=false;
 job_.Job := job;
 jparser := TJobParser.Create(plugman_, meth_, res_, frontman_, job_, 1);
 jparser.parse();
 jparser.Free;
end;

procedure TTestJobParser.TestJobParserFloats;
var float   : TStkFloat;
begin
 execJob('1,1,add');
 popFloat(float, job_.stack);
 AssertEquals('1+1 is', 2, float);

 execJob('2,3,mul');
 popFloat(float, job_.stack);
 AssertEquals('2*3 is', 6, float);

 execJob('49, sqrt');
 popFloat(float, job_.stack);
 AssertEquals('sqrt of 49 is', 7, float);
end;

procedure TTestJobParser.TestJobParserException;
begin
  //TODO: throws an Access violation, find why
  //execJob('testexception');
  //logger_.log(LVL_SEVERE, job_.stack.error.errorArg);
  //AssertEquals('Exception inside plugin', true, job_.stack.error.errorId=PLUGIN_THREW_EXCEPTION_ID);
end;

procedure TTestJobParser.SetUp;
var path, extension : String;
    error           : TStkError;
begin
 clearError(error);
 path            := ExtractFilePath(ParamStr(0));
 extension       := 'dll';
 logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log');
 plugman_        := TPluginManager.Create(path+PathDelim+'plugins'+PathDelim+'lib', extension, logger_);
 plugman_.loadAll(error);
 meth_           := TMethodController.Create();
 res_            := TResultCollector.Create();
 frontman_       := TFrontendManager.Create();
 spec_           := TSpecialCommand.Create(plugman_, meth_, res_, frontman_);
 job_            := TJob.Create();
end;

procedure TTestJobParser.TearDown;
begin
 spec_.Free;
 frontman_.Free;
 res_.Free;
 meth_.Free;
 plugman_.Free;
 logger_.Free;
 job_.Free;
end;

initialization

  RegisterTest(TTestJobParser);
end.                         
