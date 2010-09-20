unit parser;

interface

uses argretrievers, stacks, pluginmanager, methodcontrollers;

type TGPUParser = class(TObject);
 public 
   constructor Create(var plugman : TPluginManager; var meth : TMethodController; var job : TJob; threadId : Longint);
   destructor Destroy();
   
   function parse() : Boolean; overload;  
   function parse(jobStr : String; var stk : TStack; var error : TError) : Boolean; overload;
   
 private
   plugman_        : TPluginManager;
   methController_ : TMethodController;
   thrdId_         : Longint;
   job_            : TJob;

   function maxStackReached(var stk : TStack; var error : TError) : Boolean; 
end;


implementation

constructor TGPUParser.Create(var plugman : TPluginManager; var meth : TMethodController; var job : TJob; threadId : Longint);
begin
  inherited Create();
  plugMan_ := plugman;
  methController_ := meth;
  job_ := job;
  thrdId_ := threadId;
end;

destructor TGPUParser.Destroy();
begin
  inherited;
end;


function TGPUParser.maxStackReached(var stk : TStack; var error : TError) : Boolean; 
begin
 Result := false;
 if (stk.Idx>MAX_STACK_PARAM) then
       begin
         Result := true;
         error.errorID := TOO_MANY_ARGUMENTS_ID;
         error.errorMsg := TOO_MANY_ARGUMENTS;
         error.errorArg := '';
         Dec(stk.Idx);
       end;
end;


procedure TGPUParser.parse() : Boolean;; overload;
begin
   Result := parse(job_.job, job_.Stack, job_.error);
   job_.hasError := (job._error.ErrorId>0);
end;

procedure TGPUParser.parse(jobStr : String; var stk : TStack; var error : TError); overload;
var arg    : TGPUArg;
    argRetriever : TArgRetriever;
    resexpr,
	funcexists,
    hasErrors    : Boolean;
	pluginName   : String;
begin
  Result := False;
  argRetriever := TArgRetriever.Create(jobStr);
  hasErrors := false;
  
  while (argRetriever_.hasArguments() and (not hasErrors)) do
     begin
       arg := getArgument(error);
       case arg.argType of
            GPU_ERROR :  hasErrors := true; // the error structure contains the error
            GPU_FLOAT : // a float was detected
                   begin
                     Inc(stk.Idx);
                     hasErrors := maxStackReached(stk, error) then 
                     if not hasErrors then
                        begin                     
                         Stk.Stack[stk.Idx] := arg.argvalue;
                         Stk.StrStack[stk.Idx] := '';
                        end; 
                   end;
            GPU_STRING : // a string was detected
                   begin
                     Inc(stk.Idx);
                     hasErrors := maxStackReached(stk, error) then 
                     if not hasErrors then
                           begin 
                             Stk.Stack[stk.Idx] := INF;
                             Stk.StrStack[stk.Idx] := arg.argstring;
                           end;  
                   end;
             GPU_EXPRESSION :
                   begin
                     // we found an expression, we need to recursively call this method
                     resexpr := parse(arg.argstring, stk, error);
                     hasErrors := not resexpr;
                   end;     
             GPU_CALL :
                   begin
                     funcexists := plugman_.method_exists(arg.argstring, pluginName, error);
					 haserrors := not funcexists;
					 if funcexists then
					      begin
						    methController_.registerMethodCall(arg.argstring, pluginName, thrdID_);
							try
							  resexpr := plugman_.method_execute(arg.argstring, pluginName, error);
							  hasErrors := not resexpr;
							except (Exception e)
							  error.errorID := PLUGIN_THREW_EXCEPTION_ID;
							  error.errorMsg := PLUGIN_THREW_EXCEPTION;
							  error.errorArg := e.Message;
							  hasErrors := true;
							end;
							methController_.unregisterMethodCall(thrdID_);
						  end;

                   end;				   
       
       end; // case
     
     end; // while
   
   argRetriever.Free;
   Result := not hasErrors;
end;

end.