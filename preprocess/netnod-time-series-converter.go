package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
)

type BitOut struct {
	Time  int64   `json:"time"`
	Value float64 `json:"value"`
}

type Data struct {
	BitsOut []BitOut `json:"bits_out"`
}

func main() {
	inputFile := flag.String("input", "", "Path to the input JSON file")
	outputFile := flag.String("output", "", "Path to the output CSV file")
	flag.Parse()

	if *inputFile == "" || *outputFile == "" {
		fmt.Println("Usage: json_to_csv --input <input_file.json> --output <output_file.csv>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	jsonFile, err := os.Open(*inputFile)
	if err != nil {
		log.Fatalf("Failed to open JSON file '%s': %v", *inputFile, err)
	}
	defer jsonFile.Close()

	var data Data
	decoder := json.NewDecoder(jsonFile)
	if err := decoder.Decode(&data); err != nil {
		log.Fatalf("Failed to decode JSON data: %v", err)
	}

	csvFile, err := os.Create(*outputFile)
	if err != nil {
		log.Fatalf("Failed to create CSV file '%s': %v", *outputFile, err)
	}
	defer csvFile.Close()

	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	header := []string{"date", "bps"}
	if err := writer.Write(header); err != nil {
		log.Fatalf("Failed to write header to CSV: %v", err)
	}

	for _, item := range data.BitsOut {
		convertedTime := time.Unix(item.Time, 0).UTC().Format("200601021504")
		record := []string{
			convertedTime,
			strconv.FormatFloat(item.Value, 'f', -1, 64),
		}
		if err := writer.Write(record); err != nil {
			log.Fatalf("Failed to write record to CSV: %v", err)
		}
	}

	log.Printf("Convertion completed! CSV file saved at %s", *outputFile)
}
