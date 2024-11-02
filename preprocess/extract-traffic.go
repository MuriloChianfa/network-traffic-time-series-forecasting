package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	nfdump "github.com/phaag/go-nfdump"
)

const ShellToUse = "bash"

func Shellout(command string) (string, string, error) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd := exec.Command(ShellToUse, "-c", command)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

func main() {
	// Interface index of Vlan33, our main uplink
	//interfaceIndex := uint32(15)

	// Setup the target output CSV file
	csvFileName := "rt-hw-ne8k-link-level3-bps-inbound-from-20-to-26-of-october-2024.csv"
	csvFile, err := os.Create(csvFileName)
	if err != nil {
		fmt.Println("Error creating CSV file:", err)
		return
	}
	defer csvFile.Close()

	writer := csv.NewWriter(csvFile)
	defer writer.Flush()
	writer.Write([]string{"date", "bps"})

	baseDir := "10"

	for day := 20; day <= 26; day++ {
		dayStr := fmt.Sprintf("%02d", day)
		dayDir := filepath.Join(baseDir, dayStr)

		// Check if the directory exists
		if _, err := os.Stat(dayDir); os.IsNotExist(err) {
			fmt.Printf("Directory %s does not exist. Skipping.\n", dayDir)
			continue
		}

		// Get all nfcapd.* files in the directory
		files, err := filepath.Glob(filepath.Join(dayDir, "nfcapd.*"))
		if err != nil {
			fmt.Println("Error finding nfcapd files:", err)
			return
		}

		for _, file := range files {
			info, err := os.Stat(file)
			if err != nil {
				fmt.Println("Error stating file:", err)
				continue
			}

			if info.IsDir() {
				fmt.Println("Is a dir, ignoring!")
				continue
			}

			fileName := info.Name()
			dateStr := strings.TrimPrefix(fileName, "nfcapd.")

			fmt.Println("Currently working on: " + dayDir + "/" + fileName)

			Shellout("nfdump -r " + dayDir + "/" + fileName + " -w " + dayDir + "/" + fileName + "-new")

			// nffile reader, thanks phaag for your great effort
			nffile := nfdump.New()
			if err := nffile.Open(dayDir + "/" + fileName + "-new"); err != nil {
				fmt.Printf("Failed to open nf file: %v\n", err)
				os.Exit(255)
			}

			var totalBytes uint64 = 0

			if recordChannel, err := nffile.AllRecords().Get(); err != nil {
				fmt.Printf("Failed to process flows: %v\n", err)
			} else {
				for record := range recordChannel {
					if flowMisc := record.FlowMisc(); flowMisc != nil {
						//if flowMisc.Input == interfaceIndex {
						if cntFlow := record.GenericFlow(); cntFlow != nil {
							totalBytes += cntFlow.InBytes
						}
						//}
					}
				}
			}

			nffile.Close()

			// Convert the total bytes to bits per second using timeslot of 5 minutes
			totalBits := totalBytes * 8    // Converting Bytes to bits
			intervalSeconds := uint64(300) // 300 seconds = 5 minutes
			bps := totalBits / intervalSeconds

			// Write resylts to CSV file
			writer.Write([]string{dateStr, strconv.FormatUint(bps, 10)})

			Shellout("rm " + dayDir + "/" + fileName + "-new")
		}
	}

	fmt.Printf("Preprocess completed, CSV file '%s' has been created successfully!\n", csvFileName)
}
