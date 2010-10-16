unit testresultcollectors;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  resultcollectors, stacks;

type

  TTestResultCollector= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestResultCollector;
  private
    res_ : TResultCollector;

    procedure registerFloat(jobId : String; float : TStkFloat);
  end; 

implementation

procedure TTestResultCollector.registerFloat(jobId : String; float : TStkFloat);
var stk : TStack;
begin
   clearStk(stk);
   pushFloat(float, stk);
   res_.registerResult(jobId, stk);
end;

procedure TTestResultCollector.TestResultCollector;
var coll : TResultCollection;
begin
 registerFloat('a', 1);
 registerFloat('a', 2);
 registerFloat('a', 3);
 AssertEquals('Job a registered on slot 1', 1 , res_.findJobId('a'));
 res_.getResultCollection('a', coll);

 AssertEquals('Number of results is', 3, coll.N);
 AssertEquals('Sum of results is', 6, coll.sum);
 AssertEquals('Average of results is', 2, coll.avg);
 AssertEquals('Number of floats', 3, coll.N_float);
 AssertEquals('Variance of results is ', 2/3, coll.variance);
 AssertEquals('Standard deviation of results is ', sqrt(2/3), coll.stddev);

end;

procedure TTestResultCollector.SetUp; 
begin
  res_ := TResultCollector.Create();
end; 

procedure TTestResultCollector.TearDown; 
begin
  res_.Free;
end; 

initialization

  RegisterTest(TTestResultCollector); 
end.

