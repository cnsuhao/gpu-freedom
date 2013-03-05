unit lockfiles;
{ $DEFINE DEBUG}
interface

uses Sysutils;

type TLockFile = class(TObject)
  public
    constructor Create(path, filename : String);
    destructor Destroy;

    procedure createLF;
    function  exists : Boolean;
    procedure delete;

  private
    fullname_ : String;
    F_        : TextFile;
end;

implementation

constructor TLockFile.Create(path, filename : String);
begin
  inherited Create;
  fullname_ := path+pathDelim+filename;

  {$IFDEF DEBUG}WriteLn('Creating TLockfile with definition '+fullname_);{$ENDIF}

  if not DirectoryExists(path) then
    MkDir(path);
end;

destructor TLockFile.Destroy;
begin
  inherited destroy;
end;


procedure TLockFile.createLF;
begin
 {$IFDEF DEBUG}WriteLn('CreateLF ('+fullname_+')');{$ENDIF}
 if not exists then
    begin
     {$IFDEF DEBUG}WriteLn('Creating file ('+fullname_+')');{$ENDIF}
     AssignFile(F_, fullname_);
     Rewrite(F_);
     CloseFile(F_);
    end;
end;


function TLockFile.exists : Boolean;
begin
 Result := FileExists(fullname_);

 {$IFDEF DEBUG}
 if Result then
      WriteLn('Lockfile exists ('+fullname_+')')
 else
      WriteLn('Lockfile does not exist ('+fullname_+')')
 {$ENDIF}
end;

procedure TLockFile.delete;
begin
 if exists then
  begin
   DeleteFile(fullname_);
   {$IFDEF DEBUG}WriteLn('Deleting lockfile ('+fullname_+')'){$ENDIF}
  end;
end;



end.
