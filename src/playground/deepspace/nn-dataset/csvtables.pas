unit csvtables;
{(c) by 2016 HB9TVM an the GPU team. Source code is under GPL}
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, utils, quicksort, Dialogs;

const
     MAX_COLUMNS = 128; // if a table has so many columns, consider
                         // refactoring :-D

type TCSVTable = class(TObject)
   public // we know what we are doing with our datastructure anyway
     filename_,
     tablename_,
     header_,
     separator_   : String;

     fields_      : Array[1..MAX_COLUMNS] of AnsiString;
     totalfields_ : Longint;

     totalrows_      : Longint;


     tablemem_       : Array of AnsiString;
     tablememFloat   : Array of Array of Extended; // dynamic array containing floats

     constructor Create(filename, tablename, separator  : String);
     destructor  Destroy;
     function    readHeader() : AnsiString;
     function    countRows() : Longint;

     function    retrieveNFieldValue(Str : AnsiString; pos : Longint) : AnsiString;
     function    retrieveRow(pos : Longint) : AnsiString;
     procedure   loadInMemory;

   private
     F : TextFile;
     loadedInMemory_ : Boolean;

end;

implementation

constructor TCSVTable.Create(filename, tablename, separator  : String);
var column : AnsiString;
    i      : Longint;
begin
 filename_  := filename;
 tablename_ := tablename;
 separator_ := separator;
 header_    := readHeader();

 loadedInMemory_ := false;

 i:=0;
 column := Trim(ExtractParamLong(header_, separator_));
 while column<>'' do
         begin
           Inc(i);
           fields_[i] := column;
           column := Trim(ExtractParamLong(header_, separator_));
         end;
 totalfields_ := i;
 totalrows_ := countRows();

end;

function TCSVTable.readHeader() : AnsiString;
var str : AnsiString;
begin
  Result := '';
  if Trim(filename_)='' then Exit;
  if not FileExists(filename_) then
        raise Exception.Create('ERROR: filename '+filename_+' does not exist!');

  str := '';
  AssignFile(F, filename_);
  try
    Reset(F);
    ReadLn(F, Str);
  finally
    CloseFile(F);
  end;

  Result := Trim(Str);
end;

function TCSVTable.countRows() : Longint;
var count : Longint;
    str   : AnsiString;
begin
  count := 0;
  AssignFile(F, filename_);
  try
    Reset(F);
    ReadLn(F);  // skip header

    while not EOF(F) do
        begin
          ReadLn(F, Str);
          if Trim(Str)='' then continue;
          Inc(count);
        end;
  finally
    CloseFile(F);
  end;

  Result := count;
end;



function TCSVTable.retrieveNFieldValue(Str : AnsiString; pos : Longint) : AnsiString;
var i : Longint;
    value : AnsiString;
begin
  for i:=1 to pos do
     begin
       value := ExtractParamLong(Str, separator_);
     end;
  Result := value;
end;





function TCSVTable.retrieveRow(pos : Longint) : AnsiString;
var i : Longint;
    str : AnsiString;
begin
    // much faster random access if table is loaded in memory
    if loadedInMemory_ then
        begin
           Result := tablemem_[pos];
           Exit;
        end;


    AssignFile(F, filename_);

    try
      Reset(F);
      ReadLn(F, Str); // skip header

      i := 0;
      while not EOF(F) do
          begin
            Readln(F, str);
            if Trim(str)='' then continue; // we skip blank lines completely

            if i=pos then
                 begin
                    Result := Str;
                    Exit;  // Exit will call the CloseFile in the finally block!
                 end;
            Inc(i);
          end;

    finally
      CloseFile(F);
    end;


    raise Exception.Create('Internal error: Row '+IntToStr(pos)+' in filename '+filename_+' not found!');
end;

procedure TCSVTable.loadInMemory;
var i : Longint;
    str : AnsiString;
begin
  setLength(tablemem_, totalrows_+1); // we store also the header

  AssignFile(F, filename_);

    try
      Reset(F);
      ReadLn(F, Str); // skip header

      i := 0;
      while not EOF(F) do
          begin
            Readln(F, str);
            if Trim(str)='' then continue; // we skip blank lines completely

            tablemem_[i] := Str;
            Inc(i);
          end;

    finally
      CloseFile(F);
    end;

  loadedInMemory_ := true;
end;


destructor TCSVTable.Destroy;
begin
  if loadedInMemory_ then setLength(tablemem_, 0);
  inherited Destroy;
end;

end.

