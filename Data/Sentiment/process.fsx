open System
open System.IO

Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

let fop fn sent = 
  File.ReadAllLines(fn)
  |> Seq.filter (fun s -> s.StartsWith(";")|>not)
  |> Seq.filter (fun s -> s.Length>1)
  |> Seq.map (fun s -> sprintf "%s,%s" s sent)

let rnd = new Random()

let words = 
    Seq.append (fop "positive-words.txt" "1") (fop "negative-words.txt" "-1")
    |> Seq.sortBy (fun _ -> rnd.NextDouble())
 
let train,test = words |> Seq.toList |> List.partition(fun _ -> rnd.NextDouble()<0.9)

File.WriteAllLines("sentiment-train.txt",train)
File.WriteAllLines("sentiment-test.txt",test)



